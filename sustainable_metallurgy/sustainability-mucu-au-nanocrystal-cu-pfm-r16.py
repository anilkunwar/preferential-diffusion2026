"""
Au-assisted Cu-Cu diffusion bonding simulator — rectangular MPF edition (v3.4.15-sustainability)
===============================================================================
Physics: MPF + Fickian/Hart diffusion + Solute Drag + Elastoplasticity.

v3.4.15-sustainability change (NEW: quantify sustainability penalty of Poly-Cu vs. nc-Cu):
  * NEW sidebar toggle: "Bottom layer microstructure" (in 🧱 Geometry section).
    Options:
      - "nc-Cu (as-designed)" (default): uses the user's nc_gs slider value
        (e.g. 15–95 nm). This is the production configuration.
      - "Poly-Cu (reference)": forces the bottom layer to use the SAME grain
        size as the top Poly-Cu layer (poly_gs, default 150 nm). This drops
        the grain-boundary volume fraction φ_gb from ~0.10 to ~0.01.
      - "Coarse Poly-Cu (d=500 nm)": uses a fixed 500 nm grain size,
        replicating conventional bulk / annealed Cu.
    Because the Hart effective diffusivity D_eff ≈ D_gb · φ_gb, switching
    nc-Cu → Poly-Cu reduces D_eff by ~10× — making the bottom layer a
    meaningful sustainability reference.
  * NEW 🌱 Sustainability tab (after 📤 Export):
      - Bonding-quality target slider ⟨c⟩_Ωb* (default 0.06, the
        77%-of-max threshold validated by existing phase-field runs).
      - Q_gb (kJ/mol) and T_bond (K) number inputs — used for the Arrhenius
        ΔT calculator.
      - Reference Poly-Cu grain size d_ref (nm) — used to compute S_t.
      - Three primary metrics: S_t (thermal-budget multiplier), ΔT
        (temperature penalty), S_P (pressure penalty).
      - Derived engineering metrics: energy per joint, CO₂ per joint,
        throughput gain, Au utilization — with explicit nc-Cu vs. Poly-Cu
        side-by-side comparison.
      - Side-by-side ⟨c⟩_Ωb(t) plot for the current configuration and the
        Poly-Cu reference (10× time-shift is visually striking).
  * NEW module-level helpers: `_time_to_target()` and
    `compute_sustainability_metrics()`. Pure functions, no Streamlit state —
    so they can be unit-tested or called from a batch-comparison script.
  * Downstream physics is UNCHANGED: the Hart effective diffusivity, solute
    drag, spectral elastoplasticity, and GND tracking all automatically
    reflect the new φ_gb because φ_gb is computed from the order-parameter
    field, not hard-coded.

v3.4.14 change (fix top-row squashing — the REAL root cause):
  * BUG FIX: The previous v3.4.13 attempt (GridSpec with height_ratios=[1,1])
    was necessary but NOT sufficient. The real root cause was
    `apply_axis_styling()` applying `ax.set_aspect('equal')` to ALL axes
    by default — including the 4 diagnostics time-series subplots.
  * For time-series plots where x spans 0–1.5 s and y spans 0–0.05,
    `set_aspect('equal')` forces matplotlib to shrink the axes vertically
    (to preserve a 1:1 data-unit ratio) — squashing the top row, even
    when GridSpec allocates equal row heights. This is why v3.4.13 didn't
    fully fix the issue.
  * Fix: Added a `force_aspect` parameter to `apply_axis_styling()`. The
    diagnostics figure now passes `force_aspect='auto'` to ALL four
    subplots, overriding the default 'equal' aspect for time-series axes.
    Spatial-map figures (multi-field grid, overlay, vector overlay) still
    use the default 'equal' aspect — which is correct for them.
  * Also applied `force_aspect='auto'` to the Line Profile single-frame
    plot (also a 1D series, not a spatial map).
  * The function now uses `ax.set_aspect('equal', adjustable='box')` for
    spatial maps (preserves the box) and `ax.set_aspect('auto')` for
    time-series (lets matplotlib fill the available space).

v3.4.13 change (fix top-row squashing with GridSpec):
  * BUG FIX: The top-row subplots in the Diagnostics 4-panel figure were
    still being squashed even after v3.4.10's switch to constrained_layout.
    Root cause: constrained_layout sees the extra y-axis labels on the
    twinx() subplot (Mass conservation, bottom-right) and dynamically
    resizes that cell — at the expense of the top row.
  * Fix: Replaced `plt.subplots(constrained_layout=True)` with an explicit
    `GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.30)`
    that LOCKS row heights at 50/50 regardless of twin axes. The layout
    is now fully determined by the GridSpec parameters, not by a dynamic
    layout engine that gets confused by twinx().
  * Added `from matplotlib.gridspec import GridSpec` to the imports.
  * The quick size control (width × height) now properly resizes ALL
    four subplots proportionally, as the user expected.

v3.4.12 change (UI cleanup — remove duplicate size controls):
  * Removed the duplicate figure-size controls from inside the enhanced
    customization expander for the Diagnostics figure. The quick size
    control block (outside the expander) is now the ONLY place to set
    the diagnostics width/height. This eliminates the visual clutter
    and the risk of future widget-key collisions.
  * Implementation: `render_enhanced_figure_customization(size_keys=None, ...)`
    — when size_keys is None, the enhanced panel skips the figure-size
    section entirely and only shows styling controls (fonts, colors,
    markers, grid, legend, spines, ticks, text overrides).
  * Added an instant-update `st.rerun()` trigger to the quick size block:
    when the user changes the width or height and presses Enter (or
    clicks elsewhere), the figure updates immediately instead of waiting
    for the next natural rerun.
  * Design principle applied: **Progressive Disclosure** — the most
    common action (resizing) is visible immediately; advanced styling
    is grouped inside a collapsible expander.

v3.4.11 change (fix StreamlitValueBelowMinError):
  * BUG FIX: Multiple figure-size widgets (the quick size control above
    the Diagnostics figure, the enhanced panel, and the Visualization
    Settings sub-tab) all share the same `fc['diagnostics_width']` /
    `fc['diagnostics_height']` keys but had DIFFERENT min/max ranges.
    When the user typed a value outside one widget's range, the value
    got saved to the fc dict, then on the next Streamlit rerun, the OTHER
    widget (with a different min/max) would crash with
    `StreamlitValueBelowMinError` or `StreamlitValueAboveMaxError`.
  * Fix: Added a `_clamped_number_input()` helper that clamps the value
    to [min, max] BEFORE passing it to st.number_input. All size widgets
    now use this helper, so they can never receive an out-of-range value.
  * Applied to all size controls: multi_field, overlay, vector,
    diagnostics, line_profile, and the enhanced panel's size_keys.

v3.4.10 change (fix diagnostics width/height):
  * BUG FIX: The width/height controls in the Diagnostics figure
    customization panel were not working. Root cause: the conditional
    write-back (`if new_w != cached_value: fc[w_key] = new_w`) was
    silently failing because Streamlit's st.number_input with a `key=`
    parameter caches the widget value in session_state, and on some
    reruns the cached widget value matched the fc dict value (so the
    conditional skipped the write), but the fc dict hadn't actually
    been updated from the previous render. Fix: ALWAYS write the value
    (remove the conditional) in both render_inline_figure_customization
    and render_enhanced_figure_customization.
  * Added a quick size control directly above the Diagnostics figure
    (not inside an expander) so the user can change width/height
    immediately without scrolling through the full enhanced panel.
  * The quick control shows a live hint: "each subplot ≈ W/2 × H/2 in"
    with recommendations like "try 14×10 or 16×12 for less compression".

v3.4.9 change (enhanced figure customization + LIVE colormaps):
  * NEW: `render_enhanced_figure_customization()` — full per-figure
    control panel with these new controls:
    - Font sizes (Title, Labels, Ticks, Legend) — set to 0 = use theme default
    - Line styles (- -- -. :), separate for main and minor series
    - Marker styles (none, o, s, ^, v, D, +, x, *), separate for main/minor
    - Marker size and edge width
    - Per-series color pickers (5 series: Poly-Cu, nc-Cu, Avg, Max, Mass)
    - Legend: show/hide, position (11 options incl. outside), frame on/off,
      frame alpha
    - Grid: show/alpha/style (- -- : -.)/color
    - Spines (box): per-spine visibility (top/right/left/bottom) + width
    - Tick: direction (out/in/inout), width, length
    - Text overrides (title/xlabel/ylabel), figure size, axis limits,
      colorbar show/orientation
  * NEW: `render_live_colormap_picker()` — picks a colormap for EACH of
    the 13 fields directly inline. NO Save button needed — the cmap is
    applied immediately on the next render. This fixes the previous
    "cmap not working" issue where the user had to click "Save field
    settings" in the Visualization Settings tab.
  * NEW: `apply_legend_styling()` helper — re-applies legend with new
    position/frame/font-size settings after all plot() calls.
  * UPGRADED: Diagnostics 4-panel 2D line plots now honor ALL the new
    settings — line style, marker style, per-series colors, font sizes,
    legend position, grid style, spine visibility.
  * UPGRADED: Single-frame line profile now uses line_style, marker_style,
    and per-series color from the customization config.
  * Results tab (Multi-field grid mode) now shows BOTH:
    1. Live per-field colormap picker (no Save button)
    2. Enhanced per-figure customization panel
    Each in its own collapsible expander.

v3.4.8 change (auto-rebuild geometry):
  * The geometry (Voronoi grain map, Au film, nc-Cu seeds, eta fields) now
    rebuilds AUTOMATICALLY whenever any geometry-affecting slider changes
    — no Initialize button click required. The rebuild runs silently
    inside a 🔄 spinner so you see what's happening, but no success
    toast is shown on every slider drag (toast only appears when you
    click the explicit Initialize button).
  * Triggered parameters: Lx, Ly, dx, dy, poly_frac, au_frac,
    n_poly_grains, poly_gs, nc_gs, gb_w, curve_amp, use_warp, relax_st,
    seed_in.
  * When the geometry is rebuilt, any stale solver history is also
    cleared (so you don't see results from an old domain size).
  * Behavior:
      1. Change Lx from 600 → 670 nm in the sidebar.
      2. Sidebar shows "🔄 Rebuilding geometry…" for ~1-2 s.
      3. The Geometry, Results, Plotly, etc. tabs immediately reflect
         the new 670 × 360 nm domain.
  * The 🚀 Initialize button is kept for explicit re-initialization
    (e.g. if you change non-geometry parameters and want a clean
    restart).

v3.4.7 change (physical-domain sliders):
  * The sidebar now exposes **Lx** (domain width) and **Ly** (domain height)
    sliders in nm — directly, instead of asking the user to think in terms
    of grid points. The code computes Nx = round(Lx/dx) and Ny = round(Ly/dy)
    automatically, clamped to [64, 1024]. The "Actual domain" caption shows
    the real physical size after rounding.
  * dx and dy sliders are now positioned right below Lx/Ly so the user
    can tune both together: increasing Lx gives a wider physical domain,
    decreasing dx gives a finer grid (more cells).
  * Added a live **memory estimate** that warns the user when Lx×Ly
    pushes the simulation close to the OOM threshold on Streamlit Cloud
    (>1.5 GB) or the caution threshold (>500 MB).
  * Kept an "Advanced: direct Nx/Ny override" expander for users who
    want exact grid control that doesn't correspond to round(Lx/dx).
  * Examples:
      - Lx=600 nm, dx=0.8 nm → Nx=750
      - Lx=800 nm, dx=0.8 nm → Nx=1000 (≈1 GB, may OOM on small cloud)
      - Lx=600 nm, dx=0.4 nm → Nx=1500 (clamped to 1024)

v3.4.6 change (inline figure customization):
  * Added a new `render_inline_figure_customization()` helper that renders
    a compact, live-editing customization panel directly ABOVE each figure
    — no Save button needed. The panel writes directly to
    `viz_cfg['figure_customization']`, so any change is reflected on the
    next figure render (Streamlit auto-reruns on input change).
  * Inline panels are now visible on:
      - Results tab: per-mode (Multi-field grid / Overlay / Vector field)
      - Results tab: Diagnostics (4-panel time-series)
      - Line Profiles tab: Single frame + Multi-time compare
      - Plotly tab: Heatmap / 3D surface / Animation / Line profiles
  * Each inline panel is collapsible (st.expander) and includes:
      - Figure width/height (matching the figure type)
      - Title / xlabel / ylabel text overrides (blank = default)
      - Grid show/hide + alpha
      - Axis limits xmin/xmax/ymin/ymax (blank = auto)
      - Colorbar show/hide + orientation (where applicable)
      - Line width (main + minor), marker size, contour line width
      - Spine width, tick direction, tick length
  * Reason the previous (Visualization Settings → Figure Customization)
    sub-tab "did not work": it required clicking a 💾 Save button, which
    many users missed. The inline panel has no Save button. The original
    sub-tab is retained for users who prefer a single global control.

v3.4.5 change:
  * DEFAULT DOMAIN REDUCED from 800 × 400 nm → 600 × 300 nm to avoid OOM
    on Streamlit Cloud. At dx=dy=0.8 nm → Nx=750, Ny=375 (281,250 cells
    ≈ 29 MB per saved frame, ~half of the 800×400 footprint). Users
    running locally can still go up to 1024×1024.
  * Note on the OOM root cause: the memory blowup was NOT the grid itself
    but the per-grain eta arrays. With nc_grain_size=15 nm and the
    800×400 nm domain, the simulator generated ~725 nc grains → 725 eta
    fields of size (Ny, Nx) each → 725 × 4 MB = ~3 GB. Even at
    nc_grain_size=95 nm, ~20 grains × 4 MB = ~80 MB per grain phase
    field × 13 stored fields = ~1 GB per frame. Reducing the domain
    linearly cuts grain count (n_nc ∝ area) and quadratic memory.

v3.4.4 change:
  * DEFAULT DOMAIN DOUBLED to 800 nm × 400 nm. The Nx/Ny defaults are
    now 1000 / 500 (at dx=dy=0.8 nm → 800×400 nm). Same spatial
    resolution as before — just twice the simulated area. Memory per
    frame grows from ~13 MB to ~52 MB; Streamlit Cloud handles this
    comfortably.

v3.4.3 change:
  * MAX GRID RAISED from 512 → 1024 on both axes, with defaults updated
    to Nx=500 and Ny=250 (so the default domain is now ~400 nm × ~200 nm
    at dx=dy=0.8 nm, up from the previous ~205 nm × ~102 nm). Slider
    step size on N is now 4 (vs. 32 before) so any value — including
    exactly 500 and 250 — is selectable. The user asked for 500 × 250;
    that is now the out-of-the-box default.
  * A live "Domain size" caption now appears under the dx/dy sliders,
    showing the physical width × height in nm and the total cell count.
    A warning is logged if N is large (memory grows as Nx·Ny).

v3.4.2 hotfix:
  * Fix StreamlitDuplicateElementKey on key='fc_tl'. The 'Use tight_layout'
    checkbox and the 'Tick length' number_input both abbreviated to 'fc_tl'
    and collided. Renamed the checkbox to 'fc_tight_layout' and the
    number_input to 'fc_tick_length'. Audited all other `key=` arguments
    in the file — no further duplicates found.

v3.4.1 hotfix:
  * Fix StreamlitMixedNumericTypesError on the 'Marker size' and 'Tick length'
    number_input widgets — these mixed `value=float(...)` with int min/max/step.
    Now coerces the value to int via a small try/except guard so all four
    args are guaranteed to be the same Python type.

v3.4 changes:
  * TIME-UNIT SELECTOR (physics-preserving). The sidebar now exposes a
    'Time unit' dropdown (ns / μs / ms / s). All time-axis labels and
    status strings honour the chosen unit. Because D·dt/dx² is the only
    physics-relevant non-dimensional group, simply relabelling the unit
    preserves the simulation numerically (D values are interpreted as
    nm² per chosen time unit). A toggle 'Auto-rescale D when unit changes'
    additionally multiplies D_lat and D_gb by the inverse scale so that
    the SI physical interpretation is also preserved across a unit switch.
    → 'what happens in 1.5 ns' can now be displayed as 'what happens
    in 1.5 s' without changing the underlying physics.
  * EXPANDED COLORMAP LIBRARY. ColormapRegistry.BASE now contains 80+
    colormaps (rainbow, jet, turbo, inferno, plasma, magma, gist_* ,
    cubehelix, BrBG, PuOr, PiYG, tab20b/c, etc.). A per-field colormap
    picker has been added to the Visualization Settings tab, so every
    field can use a different colormap and that choice is honoured
    across multi-field grids, overlays, Plotly, animations and exports.
  * PER-FIGURE CUSTOMIZATION. New 'Figure Customization' sub-tab in
    Visualization Settings exposes: figure width/height, line/marker
    width, marker size, custom title text, custom x/y axis labels,
    grid toggle & alpha, axis limits (xmin/xmax/ymin/ymax with auto),
    colorbar show/orientation, spine/tick width and tick direction.
    These are applied to multi-field grid, overlay, vector overlay,
    diagnostics, line-profile (matplotlib + Plotly) and 3D surface.

v3.3 change:
  * Unified min/max colorbar range logic across every visualization:
    multi-field grid, overlay (contour + heatmap), vector overlay,
    Plotly heatmap, Plotly 3D surface, Plotly animation, GIF/MP4
    animation, and Compare Saved mode.
  * New FIELD_DEFAULT_RNG dictionary centralizes physical default
    ranges (c ∈ [0,1], sigma_eq ∈ auto, etc.).
  * New get_global_bounds() helper for animations ensures custom
    vmin/vmax are honored across all frames.
  * UI number inputs now seed from FIELD_DEFAULT_RNG when empty.

v3.2 change:
  * Internal key 'phi' renamed to 'c' (Au concentration); 'phi_gb'
    preserved.

v3.1 corrections (physics reverted to v2 core):
  * Plane-strain elastic moduli restored.
  * compute_local_grain_size restored.
  * compute_plasticity restored to the fixed-direction v2 flow rule.
  * Diffusion mass conservation restored (clip + rescale, no rejection).
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
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import plotly.graph_objects as go
from datetime import datetime
import json, zipfile, pickle, hashlib, traceback, warnings
from io import BytesIO, StringIO
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any

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

# v3.4.16: publication-quality sustainability figures are now INLINED later
# in this file (search for "SUSTAINABILITY MATH + PUBLICATION FIGURES").
# No external module dependency.


# ============================================================================
# SENTINEL FOR DECORATED FAILURES
# ============================================================================
class _Sentinel:
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
# ERROR HANDLING
# ============================================================================
def handle_errors(func=None, *, re_raise=False):
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
    # v3.4: 80+ colormaps covering every matplotlib family plus the 4 custom
    # ones registered by `_build_custom_colormaps` (au_conc, grain_map,
    # stress_pub, gb_hot). Reverses ('_r') are included for the most useful
    # diverging and sequential maps so the user can pick them directly.
    BASE = [
        # --- Perceptually uniform sequential ---
        'viridis', 'viridis_r', 'plasma', 'plasma_r',
        'inferno', 'inferno_r', 'magma', 'magma_r',
        'cividis', 'cividis_r', 'turbo', 'turbo_r',
        # --- Sequential (single hue) ---
        'Greys', 'Greys_r', 'Blues', 'Blues_r',
        'Greens', 'Greens_r', 'Reds', 'Reds_r',
        'Oranges', 'Oranges_r', 'Purples', 'Purples_r',
        # --- Sequential (multi-hue) ---
        'YlOrRd', 'YlOrRd_r', 'YlOrBr', 'YlOrBr_r',
        'YlGnBu', 'YlGnBu_r', 'YlGn', 'YlGn_r',
        'BuPu', 'BuPu_r', 'PuRd', 'PuRd_r',
        'BuGn', 'BuGn_r', 'PuBu', 'PuBu_r',
        'PuBuGn', 'PuBuGn_r', 'GnBu', 'GnBu_r',
        'OrRd', 'OrRd_r',
        # --- Diverging ---
        'RdBu', 'RdBu_r', 'coolwarm', 'coolwarm_r',
        'bwr', 'bwr_r', 'seismic', 'seismic_r',
        'Spectral', 'Spectral_r',
        'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
        'BrBG', 'BrBG_r', 'PiYG', 'PiYG_r',
        'PRGn', 'PRGn_r', 'PuOr', 'PuOr_r',
        'RdGy', 'RdGy_r',
        # --- Cyclic ---
        'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
        'hsv', 'jet', 'jet_r', 'rainbow', 'rainbow_r',
        'nipy_spectral', 'nipy_spectral_r', 'gist_rainbow', 'gist_rainbow_r',
        # --- Sequential (grey-ish / monotone) ---
        'binary', 'binary_r', 'gray', 'gray_r',
        'bone', 'bone_r', 'copper', 'copper_r',
        'pink', 'pink_r', 'hot', 'hot_r',
        'afmhot', 'afmhot_r', 'gist_heat', 'gist_heat_r',
        'gist_gray', 'gist_gray_r', 'gist_yarg', 'gist_yarg_r',
        # --- Miscellaneous ---
        'flag', 'prism', 'ocean', 'ocean_r',
        'terrain', 'terrain_r', 'gist_earth', 'gist_earth_r',
        'gist_ncar', 'gist_ncar_r', 'gist_stern', 'gist_stern_r',
        'CMRmap', 'CMRmap_r', 'cubehelix', 'cubehelix_r',
        'brg', 'brg_r', 'cool', 'cool_r', 'warm', 'warm_r',
        'Wistia', 'Wistia_r',
        'spring', 'spring_r', 'summer', 'summer_r',
        'autumn', 'autumn_r', 'winter', 'winter_r',
        # --- Qualitative ---
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c',
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
    'c':         'au_conc',
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
    'c':         'c (Au concentration)',
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
    'c': '', 'eta_poly': '', 'eta_au': '', 'eta_nc': '', 'grain_map': '',
    'phi_gb': '', 'D_eff': '',
    'sigma_eq': 'GPa', 'sigma_h': 'GPa', 'h': 'nm',
    'eps_p_mag': '', 'sigma_y': 'MPa', 'rho_gnd': '',
}

# NEW (v3.3): Centralized default ranges. Fields not listed here fall
# back to auto percentile-based scaling.
FIELD_DEFAULT_RNG = {
    'c':         (0.0, 1.0),
    'eta_poly':  (0.0, 1.0),
    'eta_au':    (0.0, 1.0),
    'eta_nc':    (0.0, 1.0),
    'phi_gb':    (0.0, 1.0),
    'rho_gnd':   (0.0, 1.0),
    'h':         (0.0, 100.0),
}

DISCRETE_FIELDS = {'grain_map', 'phi_gb'}

VECTOR_FIELD_SOURCES = ['c', 'sigma_h', 'sigma_eq', 'D_eff', 'phi_gb', 'rho_gnd']


# ============================================================================
# v3.4: TIME-UNIT HANDLING (physics-preserving)
# ============================================================================
# The simulation runs in *code* units. The user picks which physical unit
# to *label* the time in. Because the physics only depends on the
# non-dimensional group D·dt/dx², relabelling the unit does NOT change
# the physics:
#   - dt_code stays the same numerical value (slider shows the SAME number
#     regardless of the chosen unit; only its physical interpretation changes)
#   - D_code stays the same numerical value (interpreted as nm² per chosen
#     time unit)
#   - Displayed time = dt_code * step, labeled with the chosen unit
#
# So if dt=5e-3 with 300 steps:
#   - With unit='ns' → total = 1.5 ns (original behaviour)
#   - With unit='s'  → total = 1.5 s  (same numerical simulation,
#                                       different physical interpretation)
#
# The diffusion pattern is IDENTICAL in both cases (D·dt = const). The
# only difference is the unit label on the time axis. This is the
# physics-preserving way to make '1.5 ns' appear as '1.5 s'.
TIME_UNITS = {
    'ns': {'label': 'ns', 'long': 'nanoseconds'},
    'μs': {'label': 'μs', 'long': 'microseconds'},
    'ms': {'label': 'ms', 'long': 'milliseconds'},
    's':  {'label': 's',  'long': 'seconds'},
}

DEFAULT_TIME_UNIT = 'ns'

# v3.4.16-sustainability: SI-second conversion factors for each time unit.
# Required for energy (kWh = kW * s / 3600) and CO2 calculations — without
# this, picking time_unit='ns' would treat nanoseconds as seconds and the
# kWh value would be 1e9x too large. See code_time_to_seconds() below.
_TIME_UNIT_TO_SECONDS = {
    'ns': 1e-9, 'us': 1e-6, 'μs': 1e-6, 'ms': 1e-3, 's': 1.0,
}


def code_time_to_seconds(t_code, params):
    """Convert solver time (code units) to SI seconds using the user's
    chosen time unit. Physics-preserving: D is interpreted as nm^2 per
    chosen-unit, so the actual physical seconds scale accordingly.

    Used by every energy / CO2 / cost line in the Sustainability tab.
    """
    if t_code is None:
        return 0.0
    unit = (params or {}).get('time_unit', DEFAULT_TIME_UNIT)
    factor = _TIME_UNIT_TO_SECONDS.get(unit, 1.0)
    return float(t_code) * factor


def get_time_unit_label(params):
    """Return the human-readable time-unit string ('ns', 'μs', 'ms', 's')."""
    if params is None:
        return DEFAULT_TIME_UNIT
    return params.get('time_unit', DEFAULT_TIME_UNIT)


def get_time_scale(params):
    """v3.4: always returns 1.0. Kept for backward compatibility with
    callers that were written before the unit-relabelling refactor."""
    return 1.0


def fmt_time(t_code, params, fmt='%.4f'):
    """Format a code-time value with the chosen unit label.

    v3.4: This is just a label change — code_time IS display_time.
    """
    unit = get_time_unit_label(params)
    return f"{fmt % float(t_code)} {unit}"


def fmt_time_range(t_code_arr, params):
    """Return (array, unit_label) for plotting time axes. v3.4: no scaling."""
    unit = get_time_unit_label(params)
    return np.asarray(t_code_arr, dtype=float), unit


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
def default_field_settings():
    return {
        f: {'auto': True, 'vmin': None, 'vmax': None,
            'log': False, 'invert': False,
            'cmap': FIELD_DEFAULT_CMAPS.get(f, 'viridis')}
        for f in FIELD_DEFAULT_CMAPS.keys()
    }


def default_figure_customization():
    """v3.4: per-figure customization settings applied to all matplotlib
    figures (multi-field grid, overlay, vector overlay, diagnostics, line
    profiles) and to Plotly 3D surfaces."""
    return {
        # --- Figure size ---
        'multi_field_width':    13.5,
        'multi_field_height':   11.0,
        'overlay_width':        11.0,
        'overlay_height':        5.5,
        'vector_width':         11.0,
        'vector_height':         5.5,
        'diagnostics_width':    13.0,
        'diagnostics_height':    8.0,
        'line_profile_width':   11.0,
        'line_profile_height':   4.5,
        # --- Line / marker styling ---
        'line_width':            2.0,
        'line_width_minor':      1.5,
        'marker_size':           6,
        'marker_width':          1.0,
        'contour_line_width':    1.0,
        'quiver_width':          0.0025,
        'quiver_scale':          30.0,
        # v3.4.9: line style & marker style (for 2D line plots)
        'line_style':            '-',       # '-', '--', '-.', ':', 'none'
        'line_style_minor':      '--',      # for the "minor" series (e.g. c_top dashed)
        'marker_style':          'none',    # 'none', 'o', 's', '^', 'v', 'D', '+', 'x', '*'
        'marker_style_minor':    'none',
        # v3.4.9: per-series colors for diagnostics (Poly-Cu, nc-Cu, Avg/Max, Mass)
        'color_series_1':        '#1f77b4',  # Poly-Cu (blue)
        'color_series_2':        '#d62728',  # nc-Cu (red)
        'color_series_3':        '#2ca02c',  # Avg stress (green)
        'color_series_4':        '#2ca02c',  # Max stress (green, dashed)
        'color_series_5':        '#000000',  # Au mass (black)
        # --- Text overrides (empty string → use default) ---
        'title_override':       '',
        'xlabel_override':      '',
        'ylabel_override':      '',
        # v3.4.9: font size overrides (None = use rcParams default)
        'title_font_size':      None,
        'label_font_size':      None,
        'tick_font_size':       None,
        'legend_font_size':     None,
        # --- Grid ---
        'show_grid':            True,
        'grid_alpha':           0.3,
        'grid_linestyle':       '-',       # '-', '--', ':', '-.'
        'grid_color':           '#cccccc',
        # --- Axis limits (None means auto) ---
        'xlim':                 (None, None),
        'ylim':                 (None, None),
        # --- Colorbar ---
        'show_colorbar':        True,
        'colorbar_orientation': 'vertical',
        # --- Spines & ticks ---
        'spine_width':          1.0,
        'tick_direction':       'out',
        'tick_width':           1.0,
        'tick_length':          4,
        # v3.4.9: per-spine visibility (box on/off)
        'spine_top':            True,
        'spine_right':          True,
        'spine_left':           True,
        'spine_bottom':         True,
        # --- Legend (for 2D line plots) ---
        'show_legend':          True,
        'legend_position':      'best',  # 'best', 'upper right', 'lower left', etc.
        'legend_frameon':       True,
        'legend_framealpha':    1.0,
        # --- Misc ---
        'axis_equal':           True,
        'tight_layout':         True,
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
        'figure_customization': default_figure_customization(),
        'overlay': {
            'enabled': False,
            'base': 'sigma_eq',
            'overlay': 'phi_gb',
            'alpha': 0.5,
            'mode': 'contour',
            'n_contours': 10,
            'show_grain_ghost': False,
        },
        'vector': {
            'enabled': False,
            'source': 'c',
            'mode': 'flux',
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
    # v3.4: back-fill any missing keys for users upgrading their session
    cfg = st.session_state.viz_cfg
    defaults = default_viz_config()
    if 'figure_customization' not in cfg:
        cfg['figure_customization'] = defaults['figure_customization']
    else:
        for k, v in defaults['figure_customization'].items():
            cfg['figure_customization'].setdefault(k, v)
    # back-fill per-field cmap choice
    for fname, fset in cfg['field_settings'].items():
        fset.setdefault('cmap', FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
    return cfg


def apply_axis_styling(ax, fig_cfg, *, title=None, xlabel=None, ylabel=None,
                       title_fs=None, label_fs=None, grid_axes=None,
                       force_aspect=None):
    """v3.4: Apply common figure-customization settings to a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib Axes
    fig_cfg : dict
        The 'figure_customization' sub-dict of viz_cfg.
    title, xlabel, ylabel : str or None
        If None, the caller's previously-set text is left untouched. If
        the corresponding override field in fig_cfg is non-empty, that
        string wins over the caller's text.
    title_fs, label_fs : float or None
        Optional font-size override for the title / axis labels.
    grid_axes : matplotlib Axes or None
        If provided, grid is applied to this Axes; otherwise to `ax`.
    force_aspect : str or None
        If 'auto' or 'equal', override fig_cfg['axis_equal'] for this Axes.
        Use 'auto' for time-series plots (diagnostics) where 'equal' would
        squash the axes. Use None to defer to fig_cfg['axis_equal'].
    """
    if fig_cfg is None or ax is None:
        return
    # v3.4.9: font size resolution — caller's explicit fs wins, else fig_cfg
    if title_fs is None:
        title_fs = fig_cfg.get('title_font_size')
    if label_fs is None:
        label_fs = fig_cfg.get('label_font_size')
    tick_fs = fig_cfg.get('tick_font_size')
    # Text overrides
    if title is not None:
        override = fig_cfg.get('title_override', '').strip()
        ax.set_title(override if override else title,
                     fontsize=title_fs)
    if xlabel is not None:
        override = fig_cfg.get('xlabel_override', '').strip()
        ax.set_xlabel(override if override else xlabel,
                      fontsize=label_fs)
    if ylabel is not None:
        override = fig_cfg.get('ylabel_override', '').strip()
        ax.set_ylabel(override if override else ylabel,
                      fontsize=label_fs)
    # v3.4.9: apply tick font size
    if tick_fs is not None:
        for label_artist in ax.get_xticklabels() + ax.get_yticklabels():
            label_artist.set_fontsize(float(tick_fs))
    # Grid (v3.4.9: with style + color)
    target = grid_axes if grid_axes is not None else ax
    if fig_cfg.get('show_grid', True):
        target.grid(True,
                    alpha=float(fig_cfg.get('grid_alpha', 0.3)),
                    linestyle=fig_cfg.get('grid_linestyle', '-'),
                    color=fig_cfg.get('grid_color', '#cccccc'))
    else:
        target.grid(False)
    # Axis limits
    xlim = fig_cfg.get('xlim', (None, None))
    ylim = fig_cfg.get('ylim', (None, None))
    if xlim and (xlim[0] is not None or xlim[1] is not None):
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim and (ylim[0] is not None or ylim[1] is not None):
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    # Spines & ticks (v3.4.9: per-spine visibility)
    sw = float(fig_cfg.get('spine_width', 1.0))
    spine_visible = {
        'top':    fig_cfg.get('spine_top',    True),
        'right':  fig_cfg.get('spine_right',  True),
        'left':   fig_cfg.get('spine_left',   True),
        'bottom': fig_cfg.get('spine_bottom', True),
    }
    for spine_name, spine in ax.spines.items():
        spine.set_linewidth(sw)
        spine.set_visible(spine_visible.get(spine_name, True))
    ax.tick_params(width=float(fig_cfg.get('tick_width', 1.0)),
                   length=float(fig_cfg.get('tick_length', 4)),
                   direction=fig_cfg.get('tick_direction', 'out'))
    # v3.4.14: Aspect ratio handling.
    # - For spatial maps (2D field images): 'equal' is correct — preserves
    #   the 1:1 physical x/y ratio so a 600×300 nm domain doesn't look
    #   stretched.
    # - For time-series line plots (diagnostics): 'equal' is WRONG — it
    #   forces matplotlib to shrink the axes vertically (because x spans
    #   0–1.5 s while y spans 0–0.05), squashing the plot.
    # The caller can pass force_aspect='auto' to override the fig_cfg value
    # for time-series axes (this is what the diagnostics figure does).
    if force_aspect is not None:
        aspect_mode = force_aspect
    elif fig_cfg.get('axis_equal', True):
        aspect_mode = 'equal'
    else:
        aspect_mode = 'auto'
    try:
        if aspect_mode == 'equal':
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('auto')
    except Exception:
        pass


def apply_legend_styling(ax, fig_cfg):
    """v3.4.9: Apply legend customization (show/position/frame/font-size)
    to an Axes that already has labels. Call this AFTER all plot() calls."""
    if fig_cfg is None or ax is None:
        return
    if not fig_cfg.get('show_legend', True):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        return
    leg_pos = fig_cfg.get('legend_position', 'best')
    legend_fs = fig_cfg.get('legend_font_size')
    frameon = bool(fig_cfg.get('legend_frameon', True))
    framealpha = float(fig_cfg.get('legend_framealpha', 1.0))
    # Recreate legend with new settings
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(
        handles, labels,
        loc=leg_pos,
        frameon=frameon,
        framealpha=framealpha,
        fontsize=legend_fs)
    return legend


# ============================================================================
# v3.4.6: INLINE FIGURE CUSTOMIZATION (per-tab interactive controls)
# ============================================================================
# A compact, live-editing panel that writes directly to
# viz_cfg['figure_customization']. Use it ABOVE each figure to let the
# user tweak the figure's appearance interactively — no Save button needed.
#
# Usage:
#     render_inline_figure_customization(
#         viz_cfg,
#         size_keys=('multi_field_width', 'multi_field_height'),
#         key_prefix='res_mf',
#         show_axis_limits=True,
#         show_text_overrides=True,
#         show_grid=True,
#         show_colorbar=True,
#         show_lines=True,
#     )
#     fig = vis.create_multi_field_comparison(...)
#     st.pyplot(fig)
#
# Returns True if any control changed (so the caller can st.rerun()).
def render_inline_figure_customization(viz_cfg, *,
                                        size_keys=None,
                                        key_prefix='inl',
                                        show_axis_limits=True,
                                        show_text_overrides=True,
                                        show_grid=True,
                                        show_colorbar=True,
                                        show_lines=True,
                                        label='Figure customization'):
    """Renders a compact, inline customization panel and writes changes
    live to viz_cfg['figure_customization']. Returns True if any value
    changed during this run (so the caller can st.rerun())."""
    fc = viz_cfg.setdefault('figure_customization', default_figure_customization())
    changed = False
    default_expanded = bool(st.session_state.get(f'{key_prefix}_exp', False))

    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f'{key_prefix}_exp'] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")

        # ---- Figure size ----
        if size_keys is not None and len(size_keys) == 2:
            w_key, h_key = size_keys
            c1, c2 = st.columns(2)
            with c1:
                # v3.4.11: clamped wrapper
                new_w = _clamped_number_input(
                    "Width (in)", value=fc.get(w_key, 11.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_w",
                    help="Figure width in inches. Press Enter or click outside to apply.")
            with c2:
                new_h = _clamped_number_input(
                    "Height (in)", value=fc.get(h_key, 5.5),
                    min_value=2.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_h")
            # v3.4.10: always write (fixes width/height not updating)
            fc[w_key] = float(new_w); changed = True
            fc[h_key] = float(new_h); changed = True

        # ---- Text overrides ----
        if show_text_overrides:
            st.markdown("**Text overrides** (blank = default)")
            c1, c2, c3 = st.columns(3)
            with c1:
                t_ov = st.text_input("Title", value=fc.get('title_override', ''),
                                      key=f"{key_prefix}_ti", label_visibility="collapsed",
                                      placeholder="Title override")
            with c2:
                x_ov = st.text_input("X label", value=fc.get('xlabel_override', ''),
                                      key=f"{key_prefix}_xl", label_visibility="collapsed",
                                      placeholder="X-axis label")
            with c3:
                y_ov = st.text_input("Y label", value=fc.get('ylabel_override', ''),
                                      key=f"{key_prefix}_yl", label_visibility="collapsed",
                                      placeholder="Y-axis label")
            if t_ov != fc.get('title_override', ''):
                fc['title_override'] = t_ov; changed = True
            if x_ov != fc.get('xlabel_override', ''):
                fc['xlabel_override'] = x_ov; changed = True
            if y_ov != fc.get('ylabel_override', ''):
                fc['ylabel_override'] = y_ov; changed = True

        # ---- Grid ----
        if show_grid:
            c1, c2 = st.columns(2)
            with c1:
                new_grid = st.checkbox("Show grid",
                                        value=bool(fc.get('show_grid', True)),
                                        key=f"{key_prefix}_grid")
            with c2:
                new_ga = st.slider("Grid α", 0.0, 1.0,
                                    float(fc.get('grid_alpha', 0.3)),
                                    0.05, key=f"{key_prefix}_ga")
            if new_grid != bool(fc.get('show_grid', True)):
                fc['show_grid'] = new_grid; changed = True
            if new_ga != float(fc.get('grid_alpha', 0.3)):
                fc['grid_alpha'] = new_ga; changed = True

        # ---- Axis limits ----
        if show_axis_limits:
            xlim_cur = fc.get('xlim', (None, None))
            ylim_cur = fc.get('ylim', (None, None))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                xmin_in = st.text_input(
                    "xmin (blank=auto)",
                    value='' if xlim_cur[0] is None else str(xlim_cur[0]),
                    key=f"{key_prefix}_xmin")
            with c2:
                xmax_in = st.text_input(
                    "xmax (blank=auto)",
                    value='' if xlim_cur[1] is None else str(xlim_cur[1]),
                    key=f"{key_prefix}_xmax")
            with c3:
                ymin_in = st.text_input(
                    "ymin (blank=auto)",
                    value='' if ylim_cur[0] is None else str(ylim_cur[0]),
                    key=f"{key_prefix}_ymin")
            with c4:
                ymax_in = st.text_input(
                    "ymax (blank=auto)",
                    value='' if ylim_cur[1] is None else str(ylim_cur[1]),
                    key=f"{key_prefix}_ymax")
            def _parse(s):
                s = s.strip() if isinstance(s, str) else s
                if s is None or s == '':
                    return None
                try:
                    return float(s)
                except Exception:
                    return None
            new_xlim = (_parse(xmin_in), _parse(xmax_in))
            new_ylim = (_parse(ymin_in), _parse(ymax_in))
            if new_xlim != tuple(xlim_cur):
                fc['xlim'] = new_xlim; changed = True
            if new_ylim != tuple(ylim_cur):
                fc['ylim'] = new_ylim; changed = True

        # ---- Colorbar ----
        if show_colorbar:
            c1, c2 = st.columns(2)
            with c1:
                new_cb = st.checkbox("Show colorbar",
                                      value=bool(fc.get('show_colorbar', True)),
                                      key=f"{key_prefix}_cb")
            with c2:
                cb_o_idx = 0 if fc.get('colorbar_orientation', 'vertical') == 'vertical' else 1
                new_cbo = st.selectbox("Colorbar orient.",
                                         ['vertical', 'horizontal'],
                                         index=cb_o_idx,
                                         key=f"{key_prefix}_cbo")
            if new_cb != bool(fc.get('show_colorbar', True)):
                fc['show_colorbar'] = new_cb; changed = True
            if new_cbo != fc.get('colorbar_orientation', 'vertical'):
                fc['colorbar_orientation'] = new_cbo; changed = True

        # ---- Lines / markers ----
        if show_lines:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                new_lw = st.number_input(
                    "Line width", value=float(fc.get('line_width', 2.0)),
                    min_value=0.2, max_value=10.0, step=0.1,
                    key=f"{key_prefix}_lw")
            with c2:
                new_lwm = st.number_input(
                    "Minor lw", value=float(fc.get('line_width_minor', 1.5)),
                    min_value=0.1, max_value=8.0, step=0.1,
                    key=f"{key_prefix}_lwm")
            with c3:
                _ms_val = fc.get('marker_size', 6)
                try:
                    _ms_val = int(_ms_val) if _ms_val is not None else 6
                except (TypeError, ValueError):
                    _ms_val = 6
                new_ms = st.number_input(
                    "Marker size", value=_ms_val,
                    min_value=0, max_value=30, step=1,
                    key=f"{key_prefix}_ms")
            with c4:
                new_cw = st.number_input(
                    "Contour lw", value=float(fc.get('contour_line_width', 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_cw")
            if new_lw != float(fc.get('line_width', 2.0)):
                fc['line_width'] = new_lw; changed = True
            if new_lwm != float(fc.get('line_width_minor', 1.5)):
                fc['line_width_minor'] = new_lwm; changed = True
            if new_ms != fc.get('marker_size', 6):
                fc['marker_size'] = new_ms; changed = True
            if new_cw != float(fc.get('contour_line_width', 1.0)):
                fc['contour_line_width'] = new_cw; changed = True

        # ---- Spines & ticks (always shown) ----
        c1, c2, c3 = st.columns(3)
        with c1:
            new_sw = st.number_input(
                "Spine w", value=float(fc.get('spine_width', 1.0)),
                min_value=0.1, max_value=5.0, step=0.1,
                key=f"{key_prefix}_sw")
        with c2:
            td_idx = ['out', 'in', 'inout'].index(
                fc.get('tick_direction', 'out'))
            new_td = st.selectbox(
                "Tick dir", ['out', 'in', 'inout'],
                index=td_idx, key=f"{key_prefix}_td")
        with c3:
            _tl_val = fc.get('tick_length', 4)
            try:
                _tl_val = int(_tl_val) if _tl_val is not None else 4
            except (TypeError, ValueError):
                _tl_val = 4
            new_tl = st.number_input(
                "Tick len", value=_tl_val,
                min_value=0, max_value=20, step=1,
                key=f"{key_prefix}_tl")
        if new_sw != float(fc.get('spine_width', 1.0)):
            fc['spine_width'] = new_sw; changed = True
        if new_td != fc.get('tick_direction', 'out'):
            fc['tick_direction'] = new_td; changed = True
        if new_tl != fc.get('tick_length', 4):
            fc['tick_length'] = new_tl; changed = True

        if changed:
            st.caption("✏️ Applied live.")
        else:
            st.caption("Tweak any value — changes apply on next figure render.")

    return changed


# ============================================================================
# v3.4.9: ENHANCED INLINE FIGURE CUSTOMIZATION (2D line plots)
# ============================================================================
# Full per-figure control panel for the Results tab and Diagnostics plots.
# Includes: figure size, font sizes (title/label/tick/legend), line style
# (- -- -. :), marker style (o s ^ D x + *), per-series colors, legend
# show/position/frame, grid style+color, per-spine visibility, axis limits,
# text overrides, colorbar, ticks.
#
# Usage:
#     render_enhanced_figure_customization(
#         viz_cfg,
#         size_keys=('diagnostics_width', 'diagnostics_height'),
#         key_prefix='diag',
#         show_line_styles=True,   # show line/marker style controls
#         show_series_colors=True, # show per-series color pickers
#         show_legend=True,        # show legend controls
#         show_spines=True,        # show spine visibility controls
#         show_font_sizes=True,   # show font-size controls
#         show_colorbar=False,    # show colorbar controls (off for line plots)
#         n_series=5,              # number of per-series color pickers
#     )

LINE_STYLES = ['-', '--', '-.', ':', 'none', '']
MARKER_STYLES = ['none', 'o', 's', '^', 'v', '<', '>', 'D', 'd', '+', 'x', '*', '.', ',', '1', '2', '3', '4', '_']
LEGEND_POSITIONS = ['best', 'upper right', 'upper left', 'lower right', 'lower left',
                   'center', 'center left', 'center right', 'upper center', 'lower center',
                   'outside upper', 'outside right']
TICK_DIRECTIONS = ['out', 'in', 'inout']
GRID_LINESTYLES = ['-', '--', ':', '-.', '']


def _clamped_number_input(label, value, min_value, max_value, step=0.5,
                            key=None, help=None, format=None):
    """v3.4.11: st.number_input wrapper that clamps the value to
    [min_value, max_value] before passing it. This prevents
    StreamlitValueBelowMinError / StreamlitValueAboveMaxError when the
    value was set by a different widget with a different min/max range
    (e.g. the diagnostics_height is set by both the quick size control
    with min=3 and the enhanced panel with min=3 — if the user types
    2.5 in one widget, the other widget crashes on the next rerun)."""
    if value is None:
        value = min_value
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(min_value)
    # Clamp to the widget's min/max range
    if value < min_value:
        value = float(min_value)
    elif value > max_value:
        value = float(max_value)
    kwargs = dict(value=value, min_value=min_value, max_value=max_value, step=step)
    if key is not None:
        kwargs['key'] = key
    if help is not None:
        kwargs['help'] = help
    if format is not None:
        kwargs['format'] = format
    return st.number_input(label, **kwargs)


def render_enhanced_figure_customization(viz_cfg, *,
                                          size_keys=None,
                                          key_prefix='enh',
                                          show_line_styles=True,
                                          show_series_colors=False,
                                          show_legend=True,
                                          show_spines=True,
                                          show_font_sizes=True,
                                          show_colorbar=True,
                                          show_axis_limits=True,
                                          show_text_overrides=True,
                                          show_grid=True,
                                          n_series=5,
                                          label='Figure customization'):
    """v3.4.9: Enhanced inline panel with full control over 2D line plots
    and image plots. Writes live to viz_cfg['figure_customization']."""
    fc = viz_cfg.setdefault('figure_customization', default_figure_customization())
    default_expanded = bool(st.session_state.get(f'{key_prefix}_exp', False))

    with st.expander(f"🎨 {label}", expanded=default_expanded):
        # Keep-open toggle
        st.session_state[f'{key_prefix}_exp'] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")

        # ---------- Figure size ----------
        if size_keys is not None and len(size_keys) == 2:
            w_key, h_key = size_keys
            st.markdown("**📐 Figure size (inches)**")
            c1, c2 = st.columns(2)
            with c1:
                # v3.4.11: clamped wrapper prevents ValueBelowMin/AboveMax
                new_w = _clamped_number_input(
                    "Width", value=fc.get(w_key, 11.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_w")
            with c2:
                new_h = _clamped_number_input(
                    "Height", value=fc.get(h_key, 5.5),
                    min_value=2.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_h")
            # v3.4.10: ALWAYS write — the conditional write was causing
            # the diagnostics width/height to not update because Streamlit's
            # st.number_input with a key= parameter caches the widget value
            # in session_state, and on some reruns the cached widget value
            # matches the fc dict value (so the conditional skips the write),
            # but the fc dict hasn't actually been updated yet from a
            # previous render. Always writing guarantees consistency.
            fc[w_key] = float(new_w)
            fc[h_key] = float(new_h)

        # ---------- Font sizes ----------
        if show_font_sizes:
            st.markdown("**🔤 Font sizes** (blank = theme default)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cur = fc.get('title_font_size')
                new_tfs = st.number_input(
                    "Title", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_tfs",
                    help="0 = use theme default")
                fc['title_font_size'] = float(new_tfs) if (new_tfs is not None and new_tfs > 0) else None
            with c2:
                cur = fc.get('label_font_size')
                new_lfs = st.number_input(
                    "Labels", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=24.0, step=0.5,
                    key=f"{key_prefix}_lfs",
                    help="0 = use theme default")
                fc['label_font_size'] = float(new_lfs) if (new_lfs is not None and new_lfs > 0) else None
            with c3:
                cur = fc.get('tick_font_size')
                new_tkfs = st.number_input(
                    "Ticks", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=20.0, step=0.5,
                    key=f"{key_prefix}_tkfs",
                    help="0 = use theme default")
                fc['tick_font_size'] = float(new_tkfs) if (new_tkfs is not None and new_tkfs > 0) else None
            with c4:
                cur = fc.get('legend_font_size')
                new_lgfs = st.number_input(
                    "Legend", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=20.0, step=0.5,
                    key=f"{key_prefix}_lgfs",
                    help="0 = use theme default")
                fc['legend_font_size'] = float(new_lgfs) if (new_lgfs is not None and new_lgfs > 0) else None

        # ---------- Text overrides ----------
        if show_text_overrides:
            st.markdown("**🏷️ Text overrides** (blank = default)")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc['title_override'] = st.text_input(
                    "Title", value=fc.get('title_override', ''),
                    key=f"{key_prefix}_ti", placeholder="Title override")
            with c2:
                fc['xlabel_override'] = st.text_input(
                    "X label", value=fc.get('xlabel_override', ''),
                    key=f"{key_prefix}_xl", placeholder="X-axis label")
            with c3:
                fc['ylabel_override'] = st.text_input(
                    "Y label", value=fc.get('ylabel_override', ''),
                    key=f"{key_prefix}_yl", placeholder="Y-axis label")

        # ---------- Line styles & markers ----------
        if show_line_styles:
            st.markdown("**✏️ Line styles & markers**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ls_idx = LINE_STYLES.index(fc.get('line_style', '-')) \
                    if fc.get('line_style', '-') in LINE_STYLES else 0
                fc['line_style'] = st.selectbox(
                    "Line style (main)", LINE_STYLES, index=ls_idx,
                    key=f"{key_prefix}_ls",
                    format_func=lambda x: {'-': '— solid', '--': '–– dashed',
                                             '-.': '–· dash-dot', ':': '·· dotted',
                                             'none': 'none', '': 'none'}.get(x, x))
            with c2:
                lsm_idx = LINE_STYLES.index(fc.get('line_style_minor', '--')) \
                    if fc.get('line_style_minor', '--') in LINE_STYLES else 1
                fc['line_style_minor'] = st.selectbox(
                    "Line style (minor)", LINE_STYLES, index=lsm_idx,
                    key=f"{key_prefix}_lsm",
                    format_func=lambda x: {'-': '— solid', '--': '–– dashed',
                                             '-.': '–· dash-dot', ':': '·· dotted',
                                             'none': 'none', '': 'none'}.get(x, x))
            with c3:
                ms_idx = MARKER_STYLES.index(fc.get('marker_style', 'none')) \
                    if fc.get('marker_style', 'none') in MARKER_STYLES else 0
                fc['marker_style'] = st.selectbox(
                    "Marker (main)", MARKER_STYLES, index=ms_idx,
                    key=f"{key_prefix}_ms")
            with c4:
                msm_idx = MARKER_STYLES.index(fc.get('marker_style_minor', 'none')) \
                    if fc.get('marker_style_minor', 'none') in MARKER_STYLES else 0
                fc['marker_style_minor'] = st.selectbox(
                    "Marker (minor)", MARKER_STYLES, index=msm_idx,
                    key=f"{key_prefix}_msm")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                fc['line_width'] = st.number_input(
                    "Line width (main)", value=float(fc.get('line_width', 2.0)),
                    min_value=0.2, max_value=10.0, step=0.1,
                    key=f"{key_prefix}_lw")
            with c2:
                fc['line_width_minor'] = st.number_input(
                    "Line width (minor)", value=float(fc.get('line_width_minor', 1.5)),
                    min_value=0.1, max_value=8.0, step=0.1,
                    key=f"{key_prefix}_lwm")
            with c3:
                _ms_val = fc.get('marker_size', 6)
                try:
                    _ms_val = int(_ms_val) if _ms_val is not None else 6
                except (TypeError, ValueError):
                    _ms_val = 6
                fc['marker_size'] = st.number_input(
                    "Marker size", value=_ms_val,
                    min_value=0, max_value=30, step=1,
                    key=f"{key_prefix}_msz")
            with c4:
                fc['marker_width'] = st.number_input(
                    "Marker edge w", value=float(fc.get('marker_width', 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_mew")

        # ---------- Per-series colors ----------
        if show_series_colors:
            st.markdown("**🎨 Per-series colors** (Poly-Cu, nc-Cu, Avg, Max, Mass)")
            cols = st.columns(min(n_series, 5))
            for i in range(min(n_series, 5)):
                with cols[i]:
                    fc[f'color_series_{i+1}'] = st.color_picker(
                        f"Series {i+1}", value=fc.get(f'color_series_{i+1}', '#1f77b4'),
                        key=f"{key_prefix}_cs{i+1}")

        # ---------- Legend ----------
        if show_legend:
            st.markdown("**📋 Legend**")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc['show_legend'] = st.checkbox(
                    "Show legend", value=bool(fc.get('show_legend', True)),
                    key=f"{key_prefix}_sleg")
            with c2:
                leg_idx = LEGEND_POSITIONS.index(fc.get('legend_position', 'best')) \
                    if fc.get('legend_position', 'best') in LEGEND_POSITIONS else 0
                fc['legend_position'] = st.selectbox(
                    "Position", LEGEND_POSITIONS, index=leg_idx,
                    key=f"{key_prefix}_legpos")
            with c3:
                fc['legend_frameon'] = st.checkbox(
                    "Frame", value=bool(fc.get('legend_frameon', True)),
                    key=f"{key_prefix}_legf")
            c1, c2 = st.columns(2)
            with c1:
                fc['legend_framealpha'] = st.slider(
                    "Frame α", 0.0, 1.0,
                    float(fc.get('legend_framealpha', 1.0)),
                    0.05, key=f"{key_prefix}_legfa")

        # ---------- Grid ----------
        if show_grid:
            st.markdown("**📊 Grid**")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc['show_grid'] = st.checkbox(
                    "Show grid", value=bool(fc.get('show_grid', True)),
                    key=f"{key_prefix}_grid")
            with c2:
                fc['grid_alpha'] = st.slider(
                    "Grid α", 0.0, 1.0,
                    float(fc.get('grid_alpha', 0.3)),
                    0.05, key=f"{key_prefix}_ga")
            with c3:
                gls_idx = GRID_LINESTYLES.index(fc.get('grid_linestyle', '-')) \
                    if fc.get('grid_linestyle', '-') in GRID_LINESTYLES else 0
                fc['grid_linestyle'] = st.selectbox(
                    "Grid style", GRID_LINESTYLES, index=gls_idx,
                    key=f"{key_prefix}_gls",
                    format_func=lambda x: {'-': '— solid', '--': '–– dashed',
                                             ':': '·· dotted', '-.': '–· dash-dot',
                                             '': 'none'}.get(x, x))
            fc['grid_color'] = st.color_picker(
                "Grid color", value=fc.get('grid_color', '#cccccc'),
                key=f"{key_prefix}_gcol")

        # ---------- Axis limits ----------
        if show_axis_limits:
            st.markdown("**📏 Axis limits** (blank = auto)")
            xlim_cur = fc.get('xlim', (None, None))
            ylim_cur = fc.get('ylim', (None, None))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                xmin_in = st.text_input(
                    "xmin", value='' if xlim_cur[0] is None else str(xlim_cur[0]),
                    key=f"{key_prefix}_xmin")
            with c2:
                xmax_in = st.text_input(
                    "xmax", value='' if xlim_cur[1] is None else str(xlim_cur[1]),
                    key=f"{key_prefix}_xmax")
            with c3:
                ymin_in = st.text_input(
                    "ymin", value='' if ylim_cur[0] is None else str(ylim_cur[0]),
                    key=f"{key_prefix}_ymin")
            with c4:
                ymax_in = st.text_input(
                    "ymax", value='' if ylim_cur[1] is None else str(ylim_cur[1]),
                    key=f"{key_prefix}_ymax")
            def _parse(s):
                s = s.strip() if isinstance(s, str) else s
                if s is None or s == '':
                    return None
                try:
                    return float(s)
                except Exception:
                    return None
            fc['xlim'] = (_parse(xmin_in), _parse(xmax_in))
            fc['ylim'] = (_parse(ymin_in), _parse(ymax_in))

        # ---------- Colorbar ----------
        if show_colorbar:
            st.markdown("**.colorbar**")
            c1, c2 = st.columns(2)
            with c1:
                fc['show_colorbar'] = st.checkbox(
                    "Show colorbar", value=bool(fc.get('show_colorbar', True)),
                    key=f"{key_prefix}_cb")
            with c2:
                cb_o_idx = 0 if fc.get('colorbar_orientation', 'vertical') == 'vertical' else 1
                fc['colorbar_orientation'] = st.selectbox(
                    "Colorbar orient.", ['vertical', 'horizontal'],
                    index=cb_o_idx, key=f"{key_prefix}_cbo")

        # ---------- Spines (box) ----------
        if show_spines:
            st.markdown("**▦ Spines (box)**")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                fc['spine_top'] = st.checkbox(
                    "Top", value=bool(fc.get('spine_top', True)),
                    key=f"{key_prefix}_spt")
            with c2:
                fc['spine_right'] = st.checkbox(
                    "Right", value=bool(fc.get('spine_right', True)),
                    key=f"{key_prefix}_spr")
            with c3:
                fc['spine_left'] = st.checkbox(
                    "Left", value=bool(fc.get('spine_left', True)),
                    key=f"{key_prefix}_spl")
            with c4:
                fc['spine_bottom'] = st.checkbox(
                    "Bottom", value=bool(fc.get('spine_bottom', True)),
                    key=f"{key_prefix}_spb")
            with c5:
                fc['spine_width'] = st.number_input(
                    "Width", value=float(fc.get('spine_width', 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_sw")

        # ---------- Ticks ----------
        st.markdown("**⏱️ Ticks**")
        c1, c2, c3 = st.columns(3)
        with c1:
            td_idx = TICK_DIRECTIONS.index(fc.get('tick_direction', 'out')) \
                if fc.get('tick_direction', 'out') in TICK_DIRECTIONS else 0
            fc['tick_direction'] = st.selectbox(
                "Direction", TICK_DIRECTIONS, index=td_idx,
                key=f"{key_prefix}_td")
        with c2:
            fc['tick_width'] = st.number_input(
                "Width", value=float(fc.get('tick_width', 1.0)),
                min_value=0.1, max_value=5.0, step=0.1,
                key=f"{key_prefix}_tw")
        with c3:
            _tl_val = fc.get('tick_length', 4)
            try:
                _tl_val = int(_tl_val) if _tl_val is not None else 4
            except (TypeError, ValueError):
                _tl_val = 4
            fc['tick_length'] = st.number_input(
                "Length", value=_tl_val,
                min_value=0, max_value=20, step=1,
                key=f"{key_prefix}_tl")

        st.caption("✏️ All changes apply live — no Save button needed.")

    return True


def render_live_colormap_picker(viz_cfg, *, key_prefix='cmap',
                                  label='Per-field colormap (live)'):
    """v3.4.9: Compact LIVE per-field colormap picker — no Save button.
    Writes directly to viz_cfg['field_settings'][fname]['cmap'] so the
    next figure render uses the new cmap."""
    field_keys = list(FIELD_DEFAULT_CMAPS.keys())
    cmap_options = ColormapRegistry.list_all()
    fc = viz_cfg.setdefault('field_settings',
                              {f: {'auto': True, 'vmin': None, 'vmax': None,
                                   'log': False, 'invert': False,
                                   'cmap': FIELD_DEFAULT_CMAPS.get(f, 'viridis')}
                               for f in field_keys})
    default_expanded = bool(st.session_state.get(f'{key_prefix}_exp', False))
    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f'{key_prefix}_exp'] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")
        st.caption(
            f"Pick a colormap for each of the {len(field_keys)} fields. "
            f"Changes apply LIVE on the next render — no Save button needed. "
            f"Use the **Invert** checkbox to reverse any colormap.")
        # Show all fields in a 2-column layout
        for i in range(0, len(field_keys), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i + j < len(field_keys):
                    with col:
                        fname = field_keys[i + j]
                        fset = fc.setdefault(fname, {})
                        cur_cmap = fset.get('cmap',
                                              FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
                        cmap_idx = (cmap_options.index(cur_cmap)
                                     if cur_cmap in cmap_options else 0)
                        new_cmap = st.selectbox(
                            f"{fname}",
                            cmap_options,
                            index=cmap_idx,
                            key=f"{key_prefix}_{fname}_cmap",
                            format_func=lambda x: x,
                            help=f"{len(cmap_options)} colormaps available")
                        # Live update — no Save button
                        if new_cmap != cur_cmap:
                            fset['cmap'] = new_cmap
                        invert = st.checkbox(
                            f"Invert {fname}",
                            value=bool(fset.get('invert', False)),
                            key=f"{key_prefix}_{fname}_inv",
                            help="Reverse the colormap")
                        fset['invert'] = invert
                        # Live preview swatch
                        try:
                            swatch_cmap = ColormapRegistry.resolve(new_cmap,
                                                                     invert=invert)
                            gradient = np.linspace(0, 1, 256).reshape(1, -1)
                            fig_sw, ax_sw = plt.subplots(figsize=(3.5, 0.35),
                                                          dpi=70)
                            ax_sw.imshow(gradient, aspect='auto', cmap=swatch_cmap)
                            ax_sw.set_axis_off()
                            ax_sw.set_title(f"{fname} → {new_cmap}",
                                             fontsize=8, pad=2)
                            st.pyplot(fig_sw, clear_figure=True); plt.close(fig_sw)
                        except Exception:
                            pass
    return True


# ============================================================================
# SUBSTRATE CLASSIFICATION (data-driven sustainability architecture)
# ============================================================================
def classify_substrate(d_nm):
    """Classify substrate based on average grain size threshold.

    Rule (per the sustainability spec):
        <d> <  100 nm  ->  Nanocrystalline (nc-Cu)
        <d> >= 100 nm  ->  Polycrystalline (Poly-Cu)

    Used by the Compare Saved (Temporal) and Sustainability tabs to
    auto-label saved runs by their bottom-layer microstructure, so the
    user does not have to remember which run was nc-Cu and which was
    Poly-Cu when comparing across sessions.
    """
    d = float(d_nm)
    if d < 100.0:
        return "Nanocrystalline (nc-Cu)"
    return "Polycrystalline (Poly-Cu)"


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
# SPECTRAL ELASTICITY — v2 plane-strain moduli (restored)
# ============================================================================
class SpectralElasticSolver:
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
        self.C11_2d, self.C12_2d, self.C44_2d = C11_2d, C12_2d, C44
        lam = C12_2d
        mu  = (C11_2d - C12_2d) / 2
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
        """v2: spectral-gradient grain size."""
        s = np.sum(self.eta_list, axis=0)
        gx, gy = spectral_gradients(s, self.kx, self.ky)
        gm = np.sqrt(gx**2 + gy**2 + 1e-15)
        return np.clip(2.0 / gm, 0, 300.0)

    # ==================================================================
    # DIFFUSION — v2 mass conservation (rescale, never reject)
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
        """v2 CH step: clip + rescale, never reject."""
        sigma_h_norm = sigma_h / 1e9
        c = self.c
        df_dc = self.W * (4.0 * c**3 - 6.0 * c**2 + 2.0 * c)
        lap_c = spectral_laplacian(c, self.k2)
        mu = df_dc - self.kappa_c * lap_c + Omega * sigma_h_norm

        mu_gx, mu_gy = spectral_gradients(mu, self.kx, self.ky)
        Jx = -D_eff * mu_gx
        Jy = -D_eff * mu_gy
        div_J = spectral_divergence(Jx, Jy, self.kx, self.ky)

        c_expl = c + self.dt * div_J
        denom = 1.0 + self.dt * self.M_ref * self.kappa_c * self.k2**2
        c_new = np.real(ifft2(fft2(c_expl) / denom))

        c_new_clipped = np.clip(c_new, 0.0, 1.0)
        m_old = float(np.sum(self.c))
        m_new = float(np.sum(c_new_clipped))
        if m_old > 1e-12 and m_new > 1e-12:
            ratio = m_new / m_old
            if 0.5 < ratio < 2.0:
                c_new_clipped *= m_old / m_new
        return np.clip(c_new_clipped, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _concentration_substep(self, dt, phi_gb, D_lat, delta_D, Omega, sigma_h):
        """v2 Fickian substep: clip + rescale; no step rejection."""
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
            if 0.1 < ratio < 10.0:
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
    def compute_plasticity(self, sigma_eq, h):
        """v2: fixed-direction flow rule."""
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

        el = self.elastic.solve(exx_t, eyy_t, exy_t, app_xx, app_yy, app_xy)
        if el is FAILED or el is None:
            raise RuntimeError("Elastic solve failed")
        (sigma_eq, sxx, syy, sxy, sigma_h,
         exx_tot, eyy_tot, exy_tot) = el

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
            'c':          self.c.copy(),
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

    @handle_errors
    def extract_profile_at(self, data, x0, y0, x1, y1, n=200):
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
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
# FIELD RESCALING HELPERS
# ============================================================================
def _rescale_field(field_name, data):
    d = data.copy()
    if field_name in ('sigma_eq', 'sigma_h', 'sxx', 'syy', 'sxy'):
        d = d / 1e9
    elif field_name == 'sigma_y':
        d = d / 1e6
    return d, FIELD_UNITS.get(field_name, '')


# ============================================================================
# v3.3: UNIFIED NORM + BOUNDS
# ============================================================================
def compute_field_norm(field_name, data, fs, default_rng=None):
    """
    Return (norm, vmin, vmax).

    Priority for range selection:
      1. If fs['auto'] is False → use fs['vmin']/fs['vmax']; fall back to
         default_rng (or data min/max) when a limit is missing.
      2. If fs['auto'] is True and default_rng is provided → use it
         (e.g. c ∈ [0,1]). This makes auto mode reproducible.
      3. Otherwise (auto, no default) → 2/98 percentile (symmetric for
         sigma_h).

    Log normalization applies when fs['log'] is True.
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
        if default_rng is not None:
            vmin, vmax = float(default_rng[0]), float(default_rng[1])
        else:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
            if field_name == 'sigma_h':
                vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
    else:
        vmin = fs.get('vmin', None)
        vmax = fs.get('vmax', None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None
                    else float(np.min(finite)))
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None
                    else float(np.max(finite)))

    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-9

    norm = None
    if use_log:
        pos = finite[finite > 0]
        if pos.size > 0:
            if vmin <= 0: vmin = float(np.percentile(pos, 2))
            if vmax <= 0: vmax = float(np.percentile(pos, 98))
            if vmin >= vmax: vmin = vmax / 10.0
            try:
                norm = LogNorm(vmin=vmin, vmax=vmax)
            except Exception:
                norm = None

    return norm, vmin, vmax


def get_global_bounds(history, field_name, fs, default_rng=None):
    """
    Return (vmin, vmax) fixed across all frames of a history.
    Honors user overrides first, then default_rng, then auto percentile
    across the whole time series.
    """
    if fs is None:
        fs = {}

    if not fs.get('auto', True):
        vmin = fs.get('vmin', None)
        vmax = fs.get('vmax', None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None else 0.0)
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None else 1.0)
        return float(vmin), float(vmax)

    if default_rng is not None:
        return float(default_rng[0]), float(default_rng[1])

    vmin_all, vmax_all = [], []
    for h in history:
        if field_name not in h:
            continue
        dd, _ = _rescale_field(field_name, h[field_name])
        finite = dd[np.isfinite(dd)]
        if finite.size:
            vmin_all.append(np.percentile(finite, 2))
            vmax_all.append(np.percentile(finite, 98))
    if not vmin_all:
        return None, None

    vmin = float(np.min(vmin_all))
    vmax = float(np.max(vmax_all))
    if field_name == 'sigma_h':
        vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
    return vmin, vmax


# ============================================================================
# VECTOR FIELD COMPUTATION
# ============================================================================
def compute_vector_field(results_dict, source, mode='gradient'):
    if mode == 'principal_stress':
        sxx = results_dict.get('sxx')
        syy = results_dict.get('syy')
        sxy = results_dict.get('sxy')
        if sxx is None or syy is None or sxy is None:
            return None, None
        theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        vx = np.cos(theta)
        vy = np.sin(theta)
        return vx, vy

    f = results_dict.get(source)
    if f is None:
        return None, None

    gy, gx = np.gradient(f)
    if mode == 'flux':
        D = results_dict.get('D_eff')
        if D is None:
            D = np.ones_like(f)
        vx = -D * gx
        vy = -D * gy
    else:
        vx, vy = gx, gy
    return vx, vy


# ============================================================================
# VISUALIZER
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
        fig_cfg = style_params.get('figure_customization', {}) or {}

        fields = [
            ('c',         'c (Au concentration)',      [0, 1]),
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
        fig_w = float(fig_cfg.get('multi_field_width', 4.5 * cols))
        fig_h = float(fig_cfg.get('multi_field_height', 3.8 * rows))
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).flatten()

        show_cb = bool(fig_cfg.get('show_colorbar', True))
        cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')

        for idx, (fname, title, rng) in enumerate(avail):
            ax = axes[idx]
            d, unit = _rescale_field(fname, results_dict[fname])

            fs = field_settings.get(fname, {})
            cmap_name = (colormaps.get(fname)
                         or fs.get('cmap')
                         or style_params.get(f'{fname}_cmap')
                         or FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
            cmap = self.get_colormap(cmap_name, invert=fs.get('invert', False))

            interp = 'nearest' if fname in DISCRETE_FIELDS else 'bilinear'

            if fname == 'grain_map':
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               origin='lower', aspect='equal',
                               interpolation='nearest')
                # v3.4: text overrides + styling
                t_override = fig_cfg.get('title_override', '').strip()
                ax.set_title(t_override if t_override else title,
                             fontsize=title_fs)
                xl = fig_cfg.get('xlabel_override', '').strip() or 'x (nm)'
                yl = fig_cfg.get('ylabel_override', '').strip() or 'y (nm)'
                ax.set_xlabel(xl, fontsize=label_fs)
                ax.set_ylabel(yl, fontsize=label_fs)
                apply_axis_styling(ax, fig_cfg, grid_axes=None)
                continue

            default_rng = FIELD_DEFAULT_RNG.get(fname, rng)
            norm, vmin, vmax = compute_field_norm(fname, d, fs,
                                                   default_rng=default_rng)

            if norm is not None:
                im = ax.imshow(d, extent=self.extent, cmap=cmap, norm=norm,
                               origin='lower', aspect='equal', interpolation=interp)
            else:
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal', interpolation=interp)
            t_override = fig_cfg.get('title_override', '').strip()
            ax.set_title(t_override if t_override else title, fontsize=title_fs)
            xl = fig_cfg.get('xlabel_override', '').strip() or 'x (nm)'
            yl = fig_cfg.get('ylabel_override', '').strip() or 'y (nm)'
            ax.set_xlabel(xl, fontsize=label_fs)
            ax.set_ylabel(yl, fontsize=label_fs)
            if show_cb:
                cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction,
                                     pad=cbar_pad, orientation=cb_orient)
                if unit:
                    cbar.set_label(unit)
            apply_axis_styling(ax, fig_cfg, grid_axes=None)

        for k in range(n, len(axes)):
            axes[k].axis('off')
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    @handle_errors
    def create_overlay(self, results_dict, base_field, overlay_field,
                       alpha=0.5, mode='contour', n_contours=10,
                       show_grain_ghost=False, grain_map=None,
                       style_params=None):
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        colormaps = style_params.get('colormaps', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        if base_field not in results_dict or overlay_field not in results_dict:
            return None

        base, base_unit = _rescale_field(base_field, results_dict[base_field])
        ov, ov_unit = _rescale_field(overlay_field, results_dict[overlay_field])

        base_fs = field_settings.get(base_field, {})
        ov_fs = field_settings.get(overlay_field, {})

        base_cmap_name = (colormaps.get(base_field)
                           or base_fs.get('cmap')
                           or FIELD_DEFAULT_CMAPS.get(base_field, 'viridis'))
        base_cmap = self.get_colormap(base_cmap_name,
                                       invert=base_fs.get('invert', False))

        fig_w = float(fig_cfg.get('overlay_width', 11.0))
        fig_h = float(fig_cfg.get('overlay_height', 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if base_field in DISCRETE_FIELDS:
            im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                origin='lower', aspect='equal',
                                interpolation='nearest')
        else:
            norm, vmin, vmax = compute_field_norm(
                base_field, base, base_fs,
                default_rng=FIELD_DEFAULT_RNG.get(base_field))
            if norm is not None:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    norm=norm, origin='lower', aspect='equal',
                                    interpolation='bilinear')
            else:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    vmin=vmin, vmax=vmax,
                                    origin='lower', aspect='equal',
                                    interpolation='bilinear')

        if show_grain_ghost and grain_map is not None:
            ghost = np.where(grain_map < 0, np.nan, grain_map)
            ax.imshow(ghost, extent=self.extent,
                      cmap=ColormapRegistry.resolve('grain_map'),
                      origin='lower', aspect='equal', interpolation='nearest',
                      alpha=0.25 * alpha)

        contour_lw = float(fig_cfg.get('contour_line_width', 1.0))
        if mode == 'contour':
            ov_cmap_name = (colormaps.get(overlay_field)
                             or ov_fs.get('cmap')
                             or FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis'))
            ov_cmap = self.get_colormap(ov_cmap_name,
                                         invert=ov_fs.get('invert', False))
            x = np.linspace(self.extent[0], self.extent[1], self.Nx)
            y = np.linspace(self.extent[2], self.extent[3], self.Ny)
            X, Y = np.meshgrid(x, y)
            ov_for_contour = ov.copy()
            ov_for_contour[~np.isfinite(ov_for_contour)] = np.nan
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            try:
                if ov_vmin is None or ov_vmax is None or ov_vmax <= ov_vmin:
                    levels = n_contours
                else:
                    levels = np.linspace(ov_vmin, ov_vmax, n_contours)
                cs = ax.contour(X, Y, ov_for_contour, levels=levels,
                                cmap=ov_cmap, linewidths=contour_lw, alpha=alpha)
                ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
            except Exception as e:
                logger.warning(f"Overlay contour failed: {e}")
        else:
            ov_cmap_name = (colormaps.get(overlay_field)
                             or ov_fs.get('cmap')
                             or FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis'))
            ov_cmap = self.get_colormap(ov_cmap_name,
                                         invert=ov_fs.get('invert', False))
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            ax.imshow(ov, extent=self.extent, cmap=ov_cmap,
                      vmin=ov_vmin, vmax=ov_vmax,
                      origin='lower', aspect='equal',
                      interpolation='bilinear', alpha=alpha)

        if fig_cfg.get('show_colorbar', True):
            cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')
            cbar = plt.colorbar(im_base, ax=ax, fraction=0.046, pad=0.04,
                                  orientation=cb_orient)
            if base_unit:
                cbar.set_label(f"{base_field} ({base_unit})")

        default_title = (
            f"Overlay: {FIELD_LABELS.get(base_field, base_field)} + "
            f"{FIELD_LABELS.get(overlay_field, overlay_field)} "
            f"(mode={mode}, α={alpha:.2f})")
        apply_axis_styling(ax, fig_cfg,
                            title=default_title, title_fs=title_fs,
                            xlabel='x (nm)', ylabel='y (nm)',
                            label_fs=label_fs)
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    @handle_errors
    def create_vector_overlay(self, results_dict, source, mode='gradient',
                              density=24, color='white', scale=1.0,
                              base_field=None, style_params=None):
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        colormaps = style_params.get('colormaps', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        vx, vy = compute_vector_field(results_dict, source, mode=mode)
        if vx is None:
            return None

        fig_w = float(fig_cfg.get('vector_width', 11.0))
        fig_h = float(fig_cfg.get('vector_height', 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if base_field and base_field in results_dict:
            base, base_unit = _rescale_field(base_field, results_dict[base_field])
            base_fs = field_settings.get(base_field, {})
            base_cmap_name = (colormaps.get(base_field)
                               or base_fs.get('cmap')
                               or FIELD_DEFAULT_CMAPS.get(base_field, 'viridis'))
            base_cmap = self.get_colormap(base_cmap_name,
                                           invert=base_fs.get('invert', False))
            norm, vmin, vmax = compute_field_norm(
                base_field, base, base_fs,
                default_rng=FIELD_DEFAULT_RNG.get(base_field))
            if norm is not None:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               norm=norm, origin='lower', aspect='equal',
                               interpolation='bilinear')
            else:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal',
                               interpolation='bilinear')
            if fig_cfg.get('show_colorbar', True):
                cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             orientation=cb_orient,
                             label=f"{base_field} ({base_unit})")
        else:
            ax.set_facecolor('#111111')

        step = max(1, self.Nx // max(density, 1))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)[::step]
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)[::step]
        X, Y = np.meshgrid(x, y)
        U = vx[::step, ::step] * scale
        V = vy[::step, ::step] * scale

        mag = np.sqrt(U**2 + V**2) + 1e-12
        U_n = U / mag
        V_n = V / mag

        q_width = float(fig_cfg.get('quiver_width', 0.0025))
        q_scale = float(fig_cfg.get('quiver_scale', 30.0))
        ax.quiver(X, Y, U_n, V_n, mag, cmap='autumn', alpha=0.9,
                  scale=q_scale, width=q_width, pivot='mid')

        apply_axis_styling(ax, fig_cfg,
                            title=f"Vector field: {mode} of {source}",
                            title_fs=title_fs,
                            xlabel='x (nm)', ylabel='y (nm)',
                            label_fs=label_fs)
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0,
                              cmap_name=None, zmid=None, viz_cfg=None):
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        # v3.4: honor per-field cmap choice from viz_cfg
        chosen = (cmap_name
                  or fs.get('cmap')
                  or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        norm, vmin, vmax = compute_field_norm(
            field_name, d, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        z = d
        if fs.get('log', False):
            pos = d > 0
            z = np.where(pos, np.log10(np.maximum(d, 1e-30)), np.nan)

        if zmid is None and field_name == 'sigma_h':
            zmid = 0

        fig_cfg = viz_cfg.get('figure_customization', {})
        p_width = int(fig_cfg.get('overlay_width', 11.0) * 60)
        p_height = int(fig_cfg.get('overlay_height', 5.5) * 60)
        xl = fig_cfg.get('xlabel_override', '').strip() or 'x (nm)'
        yl = fig_cfg.get('ylabel_override', '').strip() or 'y (nm)'
        title_ov = fig_cfg.get('title_override', '').strip()
        title_str = title_ov if title_ov else f"{field_name} — Frame {frame_idx} ({chosen})"

        fig = go.Figure()
        show_cb = bool(fig_cfg.get('show_colorbar', True))
        cb_dict = dict(title=f"{field_name} {unit}") if show_cb else None
        fig.add_trace(go.Heatmap(
            z=z,
            x=np.linspace(self.extent[0], self.extent[1], self.Nx),
            y=np.linspace(self.extent[2], self.extent[3], self.Ny),
            colorscale=colorscale, zmid=zmid,
            zmin=(np.log10(vmin) if fs.get('log') and vmin and vmin > 0 else vmin),
            zmax=(np.log10(vmax) if fs.get('log') and vmax and vmax > 0 else vmax),
            colorbar=cb_dict,
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.4f}<extra></extra>'))
        fig.update_layout(title=title_str,
                          xaxis_title=xl, yaxis_title=yl,
                          yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
                          width=max(400, p_width), height=max(300, p_height),
                          template="plotly_white")
        return fig

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
        chosen = (cmap_name
                  or fs.get('cmap')
                  or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)
        X, Y = np.meshgrid(x, y)

        # v3.3: unified bounds for 3D surface colorbar
        norm, vmin, vmax = compute_field_norm(
            field_name, d, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        contours = None
        if show_contours:
            contours = dict(
                z=dict(show=True, usecolormap=True, project_z=True,
                       highlightcolor="limegreen"))

        fig = go.Figure(data=[go.Surface(
            z=d, x=X, y=Y, colorscale=colorscale, contours=contours,
            cmin=vmin, cmax=vmax,
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

    @handle_errors
    def create_multi_time_profiles(self, history, field_name, ptype='horizontal',
                                   frames=None, position_ratio=0.5,
                                   use_plotly=False, viz_cfg=None):
        if frames is None:
            frames = list(range(len(history)))
        frames = [f for f in frames if 0 <= f < len(history)]
        if not frames:
            return None

        viz_cfg = viz_cfg or {}
        fig_cfg = viz_cfg.get('figure_customization', {}) or {}

        if use_plotly:
            fig = go.Figure()
            lw = float(fig_cfg.get('line_width', 2.0))
            for f in frames:
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is FAILED or res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                fig.add_trace(go.Scatter(x=dist, y=prof, mode='lines',
                                         name=f"Frame {f}",
                                         line=dict(width=lw)))
            p_width = int(float(fig_cfg.get('line_profile_width', 11.0)) * 60)
            p_height = int(float(fig_cfg.get('line_profile_height', 4.5)) * 60)
            xl = fig_cfg.get('xlabel_override', '').strip() or 'Position (nm)'
            yl = fig_cfg.get('ylabel_override', '').strip() or f'{field_name}'
            title_ov = fig_cfg.get('title_override', '').strip()
            title_str = title_ov if title_ov else f"{field_name} — multi-time {ptype}"
            fig.update_layout(title=title_str,
                              xaxis_title=xl, yaxis_title=yl,
                              hovermode='x unified', template="plotly_white",
                              width=max(400, p_width), height=max(300, p_height))
            return fig
        else:
            fig_w = float(fig_cfg.get('line_profile_width', 11.0))
            fig_h = float(fig_cfg.get('line_profile_height', 4.5))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            cmap = plt.cm.viridis
            n = len(frames)
            lw = float(fig_cfg.get('line_width', 2.0))
            for k, f in enumerate(frames):
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is FAILED or res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                color = cmap(k / max(n - 1, 1))
                ax.plot(dist, prof, color=color, lw=lw, label=f"Frame {f}")
            apply_axis_styling(ax, fig_cfg,
                                title=f'{field_name} — multi-time {ptype}',
                                xlabel='Position (nm)',
                                ylabel=f'{field_name}')
            ax.legend(loc='best', fontsize=8)
            if fig_cfg.get('tight_layout', True):
                plt.tight_layout()
            return fig

    @handle_errors
    def create_plotly_animation(self, history, field_name, dt,
                                cmap_name=None, viz_cfg=None):
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        chosen = (cmap_name
                  or fs.get('cmap')
                  or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        # v3.3: unified bounds across the whole time series
        zmin, zmax = get_global_bounds(
            history, field_name, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        # grain_map: Plotly can't take None → fall back to data-driven bounds
        if field_name == 'grain_map':
            zmin, zmax = None, None

        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)

        frames = []
        for i, h in enumerate(history):
            d, _ = _rescale_field(field_name, h[field_name])
            frames.append(go.Frame(
                data=[go.Heatmap(z=d, x=x, y=y, colorscale=colorscale,
                                 zmin=zmin, zmax=zmax,
                                 colorbar=dict(
                                     title=f"{field_name} "
                                           f"{FIELD_UNITS.get(field_name, '')}"))],
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
            cmap_name
            or fs.get('cmap')
            or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'),
            invert=fs.get('invert', False))

        d0, unit = _rescale_field(field_name, history[0][field_name])

        # v3.3: unified bounds across the whole time series
        vmin, vmax = get_global_bounds(
            history, field_name, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))
        if field_name == 'grain_map':
            vmin, vmax = None, None

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
                    frame_out[k] = {
                        kk: (torch.from_numpy(np.asarray(vv))
                             if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()
                    }
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
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], Nx)
            y = np.linspace(extent[2], extent[3], Ny)
            X, Y = np.meshgrid(x, y)
            keys = ('c', 'eta_poly', 'eta_au', 'eta_nc', 'grain_map',
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
# SIMULATION DATABASE
# ============================================================================
class SimulationDatabase:
    MAX_SIMS = 32

    @staticmethod
    def _hash(params):
        s = json.dumps({k: v for k, v in params.items()
                        if k not in ('history', 'results', 'geom_viz',
                                     'n_steps', 'save_frequency')},
                       sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    @staticmethod
    @handle_errors
    def save(params, history, geometry_data, metadata=None, run_time=0.0,
             diagnostics=None):
        """Save a simulation to the in-memory database.

        v3.4.17 change: the new ``diagnostics`` kwarg stores the lightweight
        1D time-series (``solver.history`` — time, c_top, c_bot, au_mass,
        gb_top, gb_bot, stresses, energy). This is the bridge between the
        saved temporal data and the Sustainability / Compare-Saved-Temporal
        tabs, which previously relied on the *active in-memory solver* or
        a geometric proxy. Now they can pull measured kinetics from any
        saved run.
        """
        if 'sims' not in st.session_state:
            st.session_state.sims = {}
        sid = SimulationDatabase._hash(params)
        st.session_state.sims[sid] = {
            'id': sid, 'params': params, 'results_history': history,
            'diagnostics': diagnostics or {},   # 1D time-series for temporal comparison + sustainability
            'geometry_data': geometry_data,
            'metadata': metadata or {'created_at': datetime.now().isoformat(),
                                      'run_time': run_time},
            'created_at': datetime.now().isoformat(),
        }
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
# ANIMATION HELPER (backward compat)
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
# v3.4.15-sustainability: SUSTAINABILITY METRIC HELPERS
# ============================================================================
# Pure functions — no Streamlit state — so they can be unit-tested or
# called from a batch-comparison script. The Sustainability tab is a thin
# wrapper around `compute_sustainability_metrics()`.
#
# Physics summary
# ---------------
# The grain-boundary volume fraction is φ_gb ≈ δ / d, where δ ≈ 1.5 nm is
# the GB width and d is the grain size. The Hart effective diffusivity is
#
#     D_eff = D_lat + (D_gb - D_lat) · φ_gb ≈ D_gb · φ_gb   (since D_gb >> D_lat)
#
# Therefore D_eff scales linearly with φ_gb and inversely with d. The
# diffusion time scale t ~ L² / D_eff gives the time penalty:
#
#     S_t = t_Poly / t_nc ≈ D_eff_nc / D_eff_Poly ≈ φ_gb_nc / φ_gb_Poly ≈ d_Poly / d_nc
#
# At fixed T, t (industrial throughput), the only remaining lever is T:
#
#     D(T+ΔT) / D(T) = S_t  →  ΔT ≈ (R·T² / Q) · ln(S_t)
#
# At fixed T, t the only remaining lever is applied pressure, which drives
# the stress-drift flux J_stress = -D_eff · Ω · c · ∇(σ_h/σ0). Flux balance
# gives S_P ≈ S_t.

import numpy as _np_sus

# ============================================================================
# v3.4.16-sustainability: SUSTAINABILITY MATH + PUBLICATION FIGURES (inlined)
# ============================================================================
# Originally a separate `sustainability_figures.py` module; inlined here so
# the app is a single self-contained file. Math summary in the module-level
# docstring above.
#
# Font fallback for the figures (Noto Sans SC for CJK, DejaVu Sans for symbols).
# Font fallback — Noto Sans SC for CJK glyphs, DejaVu Sans for symbols.
# This is harmless when only Latin text is rendered and prevents the
# "missing glyph" warnings when the user uses non-ASCII characters.
# ---------------------------------------------------------------------------
try:
    _fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
try:
    _fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False


# Gas constant used by the Arrhenius calculations below.
R_GAS = 8.314                                # J/(mol*K)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def phi_gb(d, delta=1.5):
    """phi_gb ~ delta/d (clamped away from zero)."""
    return delta / np.maximum(d, 1.0)


def D_eff(d, D_lat=0.01, D_gb=50.0, delta=1.5):
    """Hart effective diffusivity (nm^2 per code-time unit)."""
    return D_lat + (D_gb - D_lat) * phi_gb(d, delta)


def S_t_formula(d_nc, d_ref, D_lat=0.01, D_gb=50.0, delta=1.5):
    """S_t = D_eff_nc / D_eff_poly (vectorised)."""
    return D_eff(d_nc, D_lat, D_gb, delta) / D_eff(d_ref, D_lat, D_gb, delta)


def effective_Q(D_lat, D_gb, phi_gb, Q_lat, Q_gb):
    """Effective Arrhenius activation energy of the parallel
    lattice + GB diffusivity.

        D_eff = D_lat + (D_gb - D_lat) * phi_gb
              = D_lat * w_lat + D_gb * w_gb       with w_lat=1, w_gb=phi_gb

    The effective Q is the D-weighted average:
        Q_eff = sum(Q_i * D_i * w_i) / sum(D_i * w_i)
    """
    w_lat = 1.0
    w_gb = float(phi_gb)
    num = Q_lat * D_lat * w_lat + Q_gb * D_gb * w_gb
    den = D_lat * w_lat + D_gb * w_gb
    return num / max(den, 1e-30)


def Delta_T_arrhenius(S_t, T, Q, exact=True):
    """Delta_T required to compensate a factor S_t in D. Vectorised.

    exact=True : exact algebraic inversion of the Arrhenius equation.
    exact=False: small-DeltaT linearisation  Delta_T ~ R*T^2/Q * ln(S_t).

    Returns 0 where S_t <= 1 (no compensation needed).
    """
    S_t = np.asarray(S_t, dtype=float)
    out = np.zeros_like(S_t)
    mask = S_t > 1.0
    if exact:
        denom = 1.0 / T - R_GAS * np.log(np.where(mask, S_t, 1.0)) / Q
        with np.errstate(divide='ignore', invalid='ignore'):
            T_new = np.where(denom > 0, 1.0 / denom, np.inf)
        out = np.where(mask, T_new - T, 0.0)
    else:
        out = np.where(mask,
                       R_GAS * T**2 / Q * np.log(np.where(mask, S_t, 1.0)),
                       0.0)
    return out


def time_at_grain_size(d, d_design, tau_nc_s, D_lat=0.01, D_gb=50.0, delta=1.5):
    """t(d) = tau_nc * D_eff(d_design) / D_eff(d)."""
    return tau_nc_s * D_eff(d_design, D_lat, D_gb, delta) \
                  / D_eff(d, D_lat, D_gb, delta)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
DEFAULT_COST = dict(
    batch_joints=100,
    furnace_kW=2.0,
    energy_price_usd_kwh=0.10,
    machine_hourly_usd=50.0,
    labor_hourly_usd=0.0,
    cu_mass_kg=1e-3,
    au_mass_kg=None,
    au_price_usd_kg=70_000.0,
    bulk_cu_usd_kg=9.0,
    nano_premium_usd_kg=50.0,
    premium_decay_nm=40.0,
    premium_ref_nm=15.0,
    nc_cu_premium_usd_kg=0.0,
)


def cost_per_joint(d, d_design, tau_nc_s, cost, D_lat=0.01, D_gb=50.0,
                   delta=1.5):
    """Energy + machine + labor + material cost per joint at grain size d.

    The nano-Cu premium decays exponentially with d (smaller powders cost
    more). Returns a dict {energy, machine, labor, au_material, cu_material,
    total}.
    """
    tau_s = time_at_grain_size(d, d_design, tau_nc_s, D_lat, D_gb, delta)
    N = cost['batch_joints']
    t_h = tau_s / 3600.0
    E_kwh = cost['furnace_kW'] * t_h

    energy = E_kwh * cost['energy_price_usd_kwh'] / N
    machine = cost['machine_hourly_usd'] * t_h / N
    labor = cost.get('labor_hourly_usd', 0.0) * t_h / N

    au_material = 0.0
    if cost.get('au_mass_kg'):
        au_material = (cost['au_mass_kg'] * cost['au_price_usd_kg']) / N

    d_scale = cost['premium_decay_nm']
    d_0 = cost['premium_ref_nm']
    premium = cost['nano_premium_usd_kg'] * np.exp(-(d - d_0) / d_scale)
    cu_material = (cost['cu_mass_kg']
                   * (cost['bulk_cu_usd_kg'] + premium)) / N

    items = {
        'energy': energy,
        'machine': machine,
        'labor': labor,
        'au_material': au_material,
        'cu_material': cu_material,
    }
    items['total'] = sum(items.values())
    return items


def break_even_premium(cost_nc, cost_poly, cu_mass_kg):
    """$/kg nc-Cu premium at which the energy saving stops justifying
    the material premium.

        break_even = (cost_poly_excl_cu - cost_nc_excl_cu) / cu_mass_kg

    cost_nc / cost_poly are the per-joint cost dicts from `cost_per_joint`.
    cu_mass_kg is the Cu mass per batch (same units as in `cost_per_joint`).
    """
    if cu_mass_kg is None or cu_mass_kg <= 0:
        return float('nan')
    poly_excl = cost_poly['total'] - cost_poly['cu_material']
    nc_excl = cost_nc['total'] - cost_nc['cu_material']
    return (poly_excl - nc_excl) / cu_mass_kg


# ---------------------------------------------------------------------------
# Composite MCDA
# ---------------------------------------------------------------------------
def composite_score(penalties, weights=None):
    """Geometric-mean MCDA.  penalties = dict(axis_name -> ratio Poly/nc).

    Returns Sigma in (0, 1]; 1 = "no penalty" (as good as Poly-Cu reference).
    A geometric mean cannot be gamed by a single dominant axis, which is
    the correct aggregation for *ratio* metrics.
    """
    if weights is None:
        weights = {'energy': 0.30, 'co2': 0.25, 'throughput': 0.20,
                   'material': 0.15, 'cost': 0.10}
    num = sum(weights[k] * np.log(max(1.0 / penalties[k], 1e-6))
              for k in weights)
    return float(np.exp(num))


# ---------------------------------------------------------------------------
# Monte-Carlo Delta_T uncertainty
# ---------------------------------------------------------------------------
def mc_DT_uncertainty(S_t, T_K=523.0, Q_gb_J=100e3, sigma_Q_J=20e3,
                      n_draws=8000, rng_seed=0):
    """Monte-Carlo 95% CI on Delta_T given Gaussian uncertainty on Q_gb.

    Q draws are clipped to [0.3, 2.0] * Q_lat to keep them physical.

    Returns dict with 'Delta_T_median' (float) and 'Delta_T_95CI' (tuple).
    """
    if S_t <= 1.0:
        return {'Delta_T_median': 0.0,
                'Delta_T_95CI': (0.0, 0.0),
                'samples': np.zeros(n_draws)}
    rng = np.random.default_rng(rng_seed)
    Q_draws = Q_gb_J + sigma_Q_J * rng.standard_normal(n_draws)
    Q_draws = np.clip(Q_draws, 0.3 * Q_gb_J, 2.0 * Q_gb_J)
    dT = Delta_T_arrhenius(S_t, T_K, Q_draws, exact=True)
    return {
        'Delta_T_median': float(np.median(dT)),
        'Delta_T_95CI': (float(np.percentile(dT, 2.5)),
                         float(np.percentile(dT, 97.5))),
        'samples': dT,
    }


# ---------------------------------------------------------------------------
# Figure 1 — S_t(d)
# ---------------------------------------------------------------------------
def fig_S_t_vs_d(params, d_nc_actual, d_ref, fname=None):
    D_lat = params.get('D_lattice', 0.01)
    D_gb = params.get('D_gb', 50.0)
    delta = params.get('gb_width_nm', 1.5)

    d_grid = np.logspace(0.0, 3.0, 500)
    S_t = S_t_formula(d_grid, d_ref, D_lat, D_gb, delta)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.loglog(d_grid, S_t, 'k-', lw=2.0,
              label=r'$S_t(d_{\rm nc})=\dfrac{D_{\rm lat}+k/d_{\rm nc}}'
                    r'{D_{\rm lat}+k/d_{\rm ref}}$')
    ax.loglog(d_grid, d_ref / d_grid, 'b--', lw=1.4, alpha=0.7,
              label=r'GB-dominant limit: $d_{\rm ref}/d_{\rm nc}$')
    ax.axhline(1.0, color='gray', ls=':', lw=1.0,
               label=r'bulk limit $S_t\to 1$')

    S_t_a = float(S_t_formula(d_nc_actual, d_ref, D_lat, D_gb, delta))
    ax.plot(d_nc_actual, S_t_a, 'ro', ms=11, zorder=5,
            label=fr'nc-Cu design: $d={d_nc_actual:.0f}$ nm, '
                  fr'$S_t={S_t_a:.1f}\times$')
    ax.plot(d_ref, 1.0, 'bs', ms=11, zorder=5,
            label=fr'Poly-Cu reference: $d={d_ref:.0f}$ nm')

    d_be = d_ref / 1.5
    ax.axvline(d_be, color='orange', ls='--', lw=1.2, alpha=0.85,
               label=fr'break-even $d={d_be:.0f}$ nm ($S_t=1.5$)')
    ax.fill_between(d_grid, 1, 1.5,
                    where=(d_grid >= d_be) & (S_t <= 1.5),
                    color='orange', alpha=0.12, label='commercially marginal')

    ax.set_xlabel(r'nc-Cu grain size $d_{\rm nc}$ (nm)')
    ax.set_ylabel(r'Thermal-budget multiplier $S_t$')
    ax.set_title(r'$S_t=\dfrac{D_{\rm eff}(d_{\rm nc})}{D_{\rm eff}(d_{\rm ref})}$, '
                 r'$k=(D_{\rm gb}-D_{\rm lat})\delta$')
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.9, 200)
    ax.grid(True, which='both', alpha=0.28)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    if fname:
        fig.savefig(fname, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Arrhenius Delta_T
# ---------------------------------------------------------------------------
def fig_Delta_T_arrhenius(params, S_t_actual, T_K=523.0, Q_gb_kJ=100.0,
                          fname=None):
    Q = Q_gb_kJ * 1e3
    Sg = np.linspace(1.01, 50.0, 400)
    dT_exact = Delta_T_arrhenius(Sg, T_K, Q, exact=True)
    dT_approx = Delta_T_arrhenius(Sg, T_K, Q, exact=False)
    dT_actual = float(Delta_T_arrhenius(S_t_actual, T_K, Q, exact=True))

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.plot(Sg, dT_exact, 'k-', lw=2.2, label='exact')
    ax.plot(Sg, dT_approx, 'b--', lw=1.6,
            label=r'$\Delta T\simeq RT^2\ln S_t/Q$')
    ax.fill_between(Sg, dT_approx, dT_exact, color='orange', alpha=0.15,
                    label='approximation error')
    ax.plot(S_t_actual, dT_actual, 'ro', ms=11, zorder=5,
            label=fr'design: $S_t={S_t_actual:.1f}$, '
                  fr'$\Delta T=+{dT_actual:.0f}$ K '
                  fr'($+{dT_actual:.0f}$ $^\circ$C)')

    ax.set_xlabel(r'Thermal-budget multiplier $S_t$')
    ax.set_ylabel(r'$\Delta T$ (K)')
    ax.set_title(fr'Arrhenius compensation ($T_0={T_K:.0f}$ K, '
                 fr'$Q_{{\rm gb}}={Q_gb_kJ:.0f}$ kJ/mol)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    if fname:
        fig.savefig(fname, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — engineering penalty bars
# ---------------------------------------------------------------------------
def fig_engineering_bars(S_t, fname=None):
    labels = ['Bonding\ntime', 'Energy\nper joint', 'CO$_2$\nper joint',
              'Throughput\n(joints/h)', r'Au util. $\eta_{\rm Au}$']
    nc = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    poly = np.array([S_t, S_t, S_t, 1.0 / S_t, S_t])
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    b1 = ax.bar(x - w / 2, nc, w, label='nc-Cu (as-designed)',
                color='#2ca02c', edgecolor='black', lw=0.6)
    b2 = ax.bar(x + w / 2, poly, w, label='Poly-Cu (reference)',
                color='#d62728', edgecolor='black', lw=0.6)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f'{b.get_height():.2f}x',
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.set_ylabel('Normalised to nc-Cu')
    ax.set_title(fr'Engineering penalty factors (Poly-Cu / nc-Cu), '
                 fr'$S_t={S_t:.1f}\times$')
    ax.legend()
    ax.grid(True, axis='y', which='both', alpha=0.3)
    if fname:
        fig.savefig(fname, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — composite MCDA radar
# ---------------------------------------------------------------------------
def fig_composite_radar(S_t, weights=None, fname=None):
    axes_lbl = ['Energy', 'CO$_2$', 'Throughput', 'Au util.', 'Cost']
    if weights is None:
        weights = {'energy': 0.30, 'co2': 0.25, 'throughput': 0.20,
                   'material': 0.15, 'cost': 0.10}
    keys = ['energy', 'co2', 'throughput', 'material', 'cost']
    pens = {k: S_t for k in keys}            # Poly/nc penalty
    sigma = composite_score(pens, weights)   # 0..1
    nc_sc = [1.0] * 5
    poly_sc = [min(1.0, 1.0 / pens[k]) for k in keys]

    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    ang += ang[:1]
    nc_sc += nc_sc[:1]
    poly_sc += poly_sc[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True),
                           constrained_layout=True)
    ax.plot(ang, nc_sc, 'o-', color='#2ca02c', lw=2.2, label='nc-Cu')
    ax.fill(ang, nc_sc, alpha=0.15, color='#2ca02c')
    ax.plot(ang, poly_sc, 's--', color='#d62728', lw=2.2, label='Poly-Cu')
    ax.fill(ang, poly_sc, alpha=0.15, color='#d62728')
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axes_lbl, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(fr'Composite score $\Sigma_{{\rm nc}}={1.0:.2f}$, '
                 fr'$\Sigma_{{\rm poly}}={sigma:.2f}$', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.02))
    if fname:
        fig.savefig(fname, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Monte-Carlo Delta_T histogram
# ---------------------------------------------------------------------------
def fig_montecarlo_DT(S_t, T_K=523.0, Q_gb_kJ=100.0, sigma_Q_kJ=20.0,
                      n=8000, seed=0, fname=None):
    Q_nom, sQ = Q_gb_kJ * 1e3, sigma_Q_kJ * 1e3
    rng = np.random.default_rng(seed)
    Q_d = np.clip(Q_nom + sQ * rng.standard_normal(n),
                  0.5 * Q_nom, 1.5 * Q_nom)
    dT = Delta_T_arrhenius(S_t, T_K, Q_d, exact=True)
    med = np.median(dT)
    lo, hi = np.percentile(dT, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.hist(dT, bins=70, color='steelblue', alpha=0.75,
            edgecolor='k', lw=0.4)
    ax.axvline(med, color='red', lw=2.2, label=fr'median = {med:+.0f} K')
    ax.axvline(lo, color='orange', ls='--', lw=1.6)
    ax.axvline(hi, color='orange', ls='--', lw=1.6,
               label=fr'95 % CI: [{lo:+.0f}, {hi:+.0f}] K')
    ax.set_xlabel(r'$\Delta T$ (K)')
    ax.set_ylabel('count')
    ax.set_title(fr'Monte-Carlo uncertainty on $\Delta T$ '
                 fr'($S_t={S_t:.1f}$, $\sigma_Q={sigma_Q_kJ:.0f}$ kJ/mol)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    if fname:
        fig.savefig(fname, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — measured kinetics (paired experiment)
# ---------------------------------------------------------------------------
def fig_kinetics(t_nc, c_nc, t_poly, c_poly, target_c,
                 t_unit='ns', fname=None,
                 nc_reached=True, poly_reached=True):
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    ax.plot(t_nc, c_nc, 'r-', lw=2.0, label='nc-Cu (measured)')
    ax.plot(t_poly, c_poly, 'b--', lw=2.0, label='Poly-Cu (measured)')
    ax.axhline(target_c, color='k', ls=':', lw=1.2,
               label=fr'target $\langle c\rangle_{{\Omega b}}^\ast={target_c:.2f}$')
    for t_arr, c_arr, col, reached in [
            (t_nc, c_nc, 'r', nc_reached),
            (t_poly, c_poly, 'b', poly_reached)]:
        c_arr = np.asarray(c_arr)
        if reached and c_arr.size and c_arr.max() >= target_c:
            i = int(np.argmax(c_arr >= target_c))
            ax.plot(t_arr[i], c_arr[i], 'o', color=col, ms=11, zorder=5)
    ax.set_xlabel(f't ({t_unit})')
    ax.set_ylabel(r'$\langle c\rangle_{\Omega b}$')
    ax.set_title('Bonding kinetics — measured curves (paired experiment)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    if fname:
        fig.savefig(fname, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — Pareto: sustainability vs cost
# ---------------------------------------------------------------------------
def fig_pareto_grain_size(params, tau_nc_s, d_nc_design, d_ref,
                          cost=None, fname=None):
    cost = cost or DEFAULT_COST
    D_lat = params.get('D_lattice', 0.01)
    D_gb = params.get('D_gb', 50.0)
    delta = params.get('gb_width_nm', 1.5)

    d_grid = np.logspace(np.log10(10), np.log10(500), 120)
    S_t = S_t_formula(d_grid, d_ref, D_lat, D_gb, delta)
    parts = [cost_per_joint(d, d_nc_design, tau_nc_s, cost,
                            D_lat, D_gb, delta) for d in d_grid]

    tot = np.array([p['total'] for p in parts])
    en = np.array([p['energy'] for p in parts])
    mc = np.array([p['machine'] for p in parts])
    ma = np.array([p['cu_material'] for p in parts])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 8.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.25]),
        constrained_layout=True)

    ax1.loglog(d_grid, S_t, 'k-', lw=2.0, label=r'$S_t(d)$ (Hart)')
    ax1.axvline(d_nc_design, color='green', ls=':', lw=1.5,
                label=fr'nc-Cu design $d={d_nc_design:.0f}$ nm')
    ax1.axvline(d_ref, color='red', ls=':', lw=1.5,
                label=fr'Poly-Cu reference $d={d_ref:.0f}$ nm')
    S_design = float(S_t_formula(d_nc_design, d_ref, D_lat, D_gb, delta))
    ax1.plot(d_nc_design, S_design, 'go', ms=11, zorder=5)
    ax1.axhline(1.5, color='orange', ls='--', lw=1.0, alpha=0.8,
                label='break-even ($S_t=1.5$)')
    ax1.set_ylabel(r'$S_t$')
    ax1.grid(True, which='both', alpha=0.28)
    ax1.set_title(r'Pareto: sustainability vs. cost')
    ax1.legend(loc='upper right', fontsize=9)

    ax2.loglog(d_grid, tot, 'k-', lw=2.2, label='total $/joint')
    ax2.loglog(d_grid, en, 'g--', lw=1.5, label='energy')
    ax2.loglog(d_grid, mc, 'b-.', lw=1.5, label='machine')
    ax2.loglog(d_grid, ma, 'm:', lw=1.5, label='material (nano premium)')
    i_min = int(np.argmin(tot))
    ax2.axvline(d_grid[i_min], color='red', ls='--', lw=1.2, alpha=0.85,
                label=fr'sweet spot $d^\star\approx{d_grid[i_min]:.0f}$ nm')
    ax2.plot(d_grid[i_min], tot[i_min], 'r*', ms=18, zorder=6)
    ax2.set_xlabel(r'bottom-layer grain size $d$ (nm)')
    ax2.set_ylabel('cost per joint (USD)')
    ax2.grid(True, which='both', alpha=0.28)
    ax2.legend()
    if fname:
        fig.savefig(fname, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Figure 8 — sensitivity heatmap S_t(d_nc, d_ref)
# ---------------------------------------------------------------------------
def fig_sensitivity_heatmap(params, fname=None,
                            d_nc_actual=None, d_ref_actual=None):
    D_lat = params.get('D_lattice', 0.01)
    D_gb = params.get('D_gb', 50.0)
    delta = params.get('gb_width_nm', 1.5)

    d_nc = np.logspace(0.0, 2.3, 140)     # 1 - 200 nm
    d_ref = np.logspace(1.0, 2.7, 140)    # 10 - 500 nm
    DNC, DREF = np.meshgrid(d_nc, d_ref)
    S = S_t_formula(DNC, DREF, D_lat, D_gb, delta)

    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    im = ax.pcolormesh(DNC, DREF, np.log10(S),
                       cmap='viridis', shading='auto')
    cs = ax.contour(DNC, DREF, S, levels=[1.5, 2, 5, 10, 20, 50],
                    colors='white', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=9, fmt=r'$%.1f\times$')
    plt.colorbar(im, ax=ax, label=r'$\log_{10}(S_t)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$d_{\rm nc}$ (nm)')
    ax.set_ylabel(r'$d_{\rm ref}$ (nm)')
    ax.set_title(r'Sustainability multiplier '
                 r'$S_t(d_{\rm nc},d_{\rm ref})$ — operating window')
    if d_nc_actual is not None and d_ref_actual is not None:
        ax.plot(d_nc_actual, d_ref_actual, 'r*', ms=18, zorder=6,
                label=fr'design point ($d_{{nc}}={d_nc_actual:.0f}$, '
                      fr'$d_{{ref}}={d_ref_actual:.0f}$)')
        ax.legend(loc='upper left')
    if fname:
        fig.savefig(fname, dpi=220)
    return fig



def _time_to_target(t_arr, c_arr, target_c):
    """Return (t_reached, reached_flag) for the first time c_arr hits
    target_c.

    v3.4.16 change: returns a TUPLE (t, reached) instead of silently
    returning NaN. The caller can then surface 'unreached' in the UI
    instead of fabricating a fallback number that would distort S_t.

    Uses linear interpolation in (t, c) space. If the maximum c_arr is
    below target_c (the simulation never reached the threshold), returns
    (inf, False). On empty input, returns (inf, False).
    """
    t_arr = _np_sus.asarray(t_arr, dtype=float)
    c_arr = _np_sus.asarray(c_arr, dtype=float)
    if t_arr.size == 0 or c_arr.size == 0:
        return float('inf'), False
    if _np_sus.max(c_arr) < target_c:
        return float('inf'), False
    # np.interp wants monotonically-increasing x — c_arr is generally
    # monotonically increasing in this kind of diffusion simulation, but
    # to be safe we sort by (c, t) and then interpolate.
    order = _np_sus.argsort(c_arr)
    c_sorted = c_arr[order]
    t_sorted = t_arr[order]
    # Remove duplicate c values (np.interp doesn't like non-unique x)
    mask = _np_sus.concatenate(([True], _np_sus.diff(c_sorted) > 0))
    c_unique = c_sorted[mask]
    t_unique = t_sorted[mask]
    if c_unique.size < 2:
        return float(t_arr[_np_sus.argmin(_np_sus.abs(c_arr - target_c))]), True
    return float(_np_sus.interp(target_c, c_unique, t_unique)), True


# effective_Q() is defined in the inlined figure block above — no wrapper needed.


def compute_cost_per_joint(*, t_seconds, furnace_kW,
                            batch_joints=100,
                            energy_price_usd_kwh=0.10,
                            machine_hourly_usd=50.0,
                            labor_hourly_usd=0.0,
                            au_mass_kg=None, au_price_usd_kg=70_000.0,
                            cu_mass_kg=None,
                            bulk_cu_usd_kg=9.0,
                            nc_cu_premium_usd_kg=0.0):
    """Return $/joint broken down by line item.

    This is the SI-seconds version — the caller MUST convert solver
    code-time to seconds via code_time_to_seconds() first. (Bug #1 fix.)
    """
    t_h = float(t_seconds) / 3600.0
    E_kwh = furnace_kW * t_h
    items = {
        'energy': E_kwh * energy_price_usd_kwh / batch_joints,
        'machine': machine_hourly_usd * t_h / batch_joints,
        'labor': labor_hourly_usd * t_h / batch_joints,
        'au_material': ((au_mass_kg or 0.0) * au_price_usd_kg) / batch_joints,
        'cu_material': ((cu_mass_kg or 0.0)
                         * (bulk_cu_usd_kg + nc_cu_premium_usd_kg)) / batch_joints,
    }
    items['total'] = sum(items.values())
    return items


def composite_sustainability_score(*, S_t, energy_ratio, co2_ratio,
                                     cost_ratio, weights=None):
    """Geometric-mean MCDA.  Returns Sigma in (0, 1]; 1 = "as good as
    Poly-Cu reference". A geometric mean cannot be gamed by a single
    dominant axis, which is the correct aggregation for *ratio* metrics.

    `*_ratio` arguments are Poly/nc penalty factors (>=1 means Poly is
    worse on that axis). Delegates to the inlined `composite_score`."""
    if weights is None:
        weights = {'energy': 0.30, 'co2': 0.25, 'throughput': 0.20,
                   'material': 0.15, 'cost': 0.10}
    penalties = {
        'energy':     max(energy_ratio, 1.0),
        'co2':        max(co2_ratio,    1.0),
        'throughput': max(S_t,          1.0),
        'material':   max(S_t,          1.0),  # Au utilisation ~ S_t
        'cost':       max(cost_ratio,   1.0),
    }
    return composite_score(penalties, weights)


def mc_sustainability_uncertainty(*, t_nc_s, t_poly_s, Q_gb_J, Q_lat_J,
                                    T_K, sigma_Q=20e3, n_draws=4000,
                                    rng_seed=0):
    """Monte-Carlo 95% CI on ΔT given Gaussian uncertainty on Q_gb.

    Q draws are clipped to [0.3, 2.0] * Q_gb to keep them physical.
    Returns dict with 'Delta_T_median', 'Delta_T_95CI', 'samples'.
    Delegates to the inlined `mc_DT_uncertainty`."""
    S_t = t_poly_s / max(t_nc_s, 1e-30)
    if S_t <= 1.0:
        return {'Delta_T_median': 0.0,
                'Delta_T_95CI': (0.0, 0.0),
                'samples': _np_sus.zeros(n_draws)}
    return mc_DT_uncertainty(S_t, T_K=T_K, Q_gb_J=Q_gb_J,
                              sigma_Q_J=sigma_Q, n_draws=n_draws,
                              rng_seed=rng_seed)


@handle_errors
def run_paired_experiment(params_base, *,
                           target_c=0.06,
                           max_steps=None,
                           convergence_window=100,
                           save_freq=None,
                           c0=None, eta0=None):
    """Run the SAME physics twice — once with bottom=nc-Cu (as-designed)
    and once with bottom=Poly-Cu (reference) — and return MEASURED
    (not estimated) quantities for both configurations.

    This is the single highest-leverage change in v3.4.16: it replaces
    the geometric time-stretch `t_poly = t_nc * S_t` with an actual
    second solver run, so S_t becomes a measured ratio rather than a
    proxy.

    Parameters
    ----------
    params_base : dict
        Base parameter set. `bottom_layer_type` and `nc_grain_size_nm`
        are overridden per-run.
    target_c : float
        Region-averaged <c>_Ωb at which the joint is considered bonded.
    max_steps : int or None
        Per-run step budget. Defaults to params_base['n_steps'] (or 300).
    convergence_window : int
        Number of saved frames that must exist before we start checking
        for the target crossing.
    save_freq : int or None
        History save frequency. Defaults to params_base['save_frequency']
        (or 10).
    c0, eta0 : array or list of arrays or None
        Initial concentration / order-parameter fields. If None, the
        solver rebuilds them from the geometry (which is what we want —
        each run gets its own geometry matching its bottom-layer choice).

    Returns
    -------
    dict with keys 'nc' and 'poly', each containing:
        t_reached     : float  — code-time units; NaN if not reached
        reached       : bool
        c_bot         : np.ndarray — full <c>_Ωb(t) trajectory
        time          : np.ndarray — time trajectory (code units)
        au_mass       : np.ndarray
        thermal_dose  : float — integral of T*dt (code units * K)
    Plus 'S_t_measured' on the 'poly' entry if both runs reached target.
    """
    if max_steps is None:
        max_steps = int(params_base.get('n_steps', 300))
    if save_freq is None:
        save_freq = int(params_base.get('save_frequency', 10))
    save_freq = max(1, save_freq)

    T_bond_K = float(params_base.get('T_bond_K', 523.0))
    dt_code = float(params_base.get('dt', 1e-3))
    poly_gs_ref = float(params_base.get('poly_grain_size_nm', 150.0))

    # Each (key, bottom_layer_type, nc_gs_override) tuple defines one run.
    # 'nc'   uses the user's nc_grain_size_nm (production design)
    # 'poly' forces the bottom layer to use the SAME grain size as the top
    #        Poly-Cu layer (so φ_gb drops from ~0.10 to ~0.01).
    run_specs = [
        ('nc',   'nc-Cu (as-designed)', None),
        ('poly', 'Poly-Cu (reference)', poly_gs_ref),
    ]

    results = {}
    for key, bottom_type, nc_gs_override in run_specs:
        p = dict(params_base)
        p['bottom_layer_type'] = bottom_type
        if nc_gs_override is not None:
            p['nc_grain_size_nm'] = float(nc_gs_override)
        # Ensure max_steps/save_freq are honoured even if the user changed
        # them in the sidebar between paired runs.
        p['n_steps'] = max_steps
        p['save_frequency'] = save_freq

        solver = AuCuBondingSolver(p)
        if c0 is not None:
            solver.c = c0.copy()
        if eta0 is not None:
            solver.eta_list = [e.copy() for e in eta0]

        t_hit, hit = float('inf'), False
        c_hist, t_hist, au_hist = [], [], []
        thermal_dose = 0.0
        for step in range(max_steps):
            r = solver.step()
            if r is FAILED or r is None:
                break
            if step % save_freq == 0 or step == max_steps - 1:
                c_hist.append(float(solver.history['c_bot'][-1]))
                t_hist.append(float(solver.history['time'][-1]))
                au_hist.append(float(solver.history['au_mass'][-1]))
            # Convergence check — only after a small warm-up so we don't
            # misfire on the first save (where c_bot is essentially zero).
            if step > 5 and len(c_hist) > convergence_window // save_freq:
                t_hit_try, hit_try = _time_to_target(
                    _np_sus.array(t_hist),
                    _np_sus.array(c_hist),
                    target_c)
                if hit_try:
                    t_hit, hit = float(t_hit_try), True
                    break
            # Cumulative thermal dose ∫ T dt (T in Kelvin, dt in code units —
            # the unit cancels in the *ratio* Poly/nc, so we keep code units
            # here and convert downstream when needed).
            thermal_dose += T_bond_K * dt_code

        # Final interpolated crossing time (in case we broke out without
        # hitting, but the trajectory did cross late — be defensive).
        if not hit and len(c_hist) > 1:
            t_hit_try, hit_try = _time_to_target(
                _np_sus.array(t_hist),
                _np_sus.array(c_hist),
                target_c)
            if hit_try:
                t_hit, hit = float(t_hit_try), True

        results[key] = {
            't_reached': float(t_hit),
            'reached': bool(hit),
            'c_bot': _np_sus.array(c_hist),
            'time': _np_sus.array(t_hist),
            'au_mass': _np_sus.array(au_hist),
            'thermal_dose': float(thermal_dose),
            'S_t_measured': float('nan'),
        }

    if results['nc']['reached'] and results['poly']['reached']:
        t_nc = max(results['nc']['t_reached'], 1e-30)
        results['poly']['S_t_measured'] = float(
            results['poly']['t_reached'] / t_nc)
    return results


@handle_errors
def sweep_grain_size(params_base, d_values, target_c,
                      max_steps=None, save_freq=None):
    """For each d in d_values, run the solver with bottom=Poly-Cu and
    grain size = d, and return (d, t_reached) pairs. Produces the t(d)
    curve that motivates the nc-Cu design choice.

    Returns a list of dicts: [{d, t_reached, reached}, ...]
    """
    if max_steps is None:
        max_steps = int(params_base.get('n_steps', 300))
    if save_freq is None:
        save_freq = int(params_base.get('save_frequency', 10))
    save_freq = max(1, save_freq)
    dt_code = float(params_base.get('dt', 1e-3))

    out = []
    for d in d_values:
        p = dict(params_base)
        p['bottom_layer_type'] = 'Poly-Cu (reference)'
        p['nc_grain_size_nm'] = float(d)
        p['n_steps'] = max_steps
        p['save_frequency'] = save_freq
        try:
            solver = AuCuBondingSolver(p)
        except Exception as e:
            logger.warning(f"sweep_grain_size: solver build failed at d={d}: {e}")
            out.append({'d': float(d), 't_reached': float('inf'),
                         'reached': False})
            continue
        t_hit, hit = float('inf'), False
        c_hist, t_hist = [], []
        for step in range(max_steps):
            r = solver.step()
            if r is FAILED or r is None:
                break
            if step % save_freq == 0 or step == max_steps - 1:
                c_hist.append(float(solver.history['c_bot'][-1]))
                t_hist.append(float(solver.history['time'][-1]))
            if step > 5 and len(c_hist) > 5:
                t_try, hit_try = _time_to_target(
                    _np_sus.array(t_hist), _np_sus.array(c_hist), target_c)
                if hit_try:
                    t_hit, hit = float(t_try), True
                    break
        if not hit and len(c_hist) > 1:
            t_try, hit_try = _time_to_target(
                _np_sus.array(t_hist), _np_sus.array(c_hist), target_c)
            if hit_try:
                t_hit, hit = float(t_try), True
        out.append({'d': float(d), 't_reached': float(t_hit),
                     'reached': bool(hit)})
    return out


def build_sustainability_report(metrics, params, *, summary_text=''):
    """Return a JSON-serializable dict with every sustainability metric,
    its 95% CI (if computed), and the full parameter set.

    Caller can st.download_button() this as .json or render it as
    Markdown via the helper below.
    """
    return {
        'version': '3.4.16',
        'params_hash': SimulationDatabase._hash(params),
        'target_c': metrics.get('target_c'),
        'S_t': metrics.get('S_t'),
        'S_t_measured': metrics.get('S_t_measured'),
        'nc_reached': metrics.get('nc_reached'),
        'poly_reached': metrics.get('poly_reached'),
        'Delta_T_gb': metrics.get('Delta_T_gb'),
        'Delta_T_lat': metrics.get('Delta_T_lat'),
        'Delta_T_95CI': metrics.get('Delta_T_95CI'),
        'Delta_T_median': metrics.get('Delta_T_median'),
        'S_P': metrics.get('S_P'),
        'cost_per_joint_nc': metrics.get('cost_nc', {}).get('total'),
        'cost_per_joint_poly': metrics.get('cost_poly', {}).get('total'),
        'cost_breakdown_nc': metrics.get('cost_nc'),
        'cost_breakdown_poly': metrics.get('cost_poly'),
        'break_even_premium_usd_kg': metrics.get('break_even_premium'),
        'break_even_grain_size_nm': metrics.get('break_even_grain_size'),
        'composite_score': metrics.get('composite_score'),
        'energy_per_joint_nc_kWh': metrics.get('E_nc_kWh'),
        'energy_per_joint_poly_kWh': metrics.get('E_poly_kWh'),
        'co2_per_joint_nc_kg': metrics.get('CO2_nc'),
        'co2_per_joint_poly_kg': metrics.get('CO2_poly'),
        'throughput_nc_per_h': metrics.get('throughput_nc'),
        'throughput_poly_per_h': metrics.get('throughput_poly'),
        'eta_Au_nc': metrics.get('eta_Au_nc'),
        'eta_Au_poly': metrics.get('eta_Au_poly'),
        'phi_gb_nc': metrics.get('phi_gb_nc'),
        'phi_gb_poly': metrics.get('phi_gb_poly'),
        'D_eff_nc': metrics.get('D_eff_nc'),
        'D_eff_poly': metrics.get('D_eff_poly'),
        'Q_eff_nc_kJmol': metrics.get('Q_eff_nc_kJmol'),
        'Q_eff_poly_kJmol': metrics.get('Q_eff_poly_kJmol'),
        'narrative': summary_text,
        'params': {k: v for k, v in params.items()
                    if not isinstance(v, _np_sus.ndarray)},
    }


def make_markdown_report(rep):
    """Render a sustainability-report dict (from build_sustainability_report)
    as a Markdown string for download."""
    L = []
    L.append("# Au-Cu Bonding — Sustainability Report")
    L.append("")
    L.append(f"**Version:** {rep.get('version')}  ")
    L.append(f"**Params hash:** `{rep.get('params_hash')}`  ")
    L.append(f"**Target ⟨c⟩*:** {rep.get('target_c')}  ")
    L.append("")
    L.append("## Primary metrics")
    L.append("")
    L.append(f"- **S_t (geometric):** {rep.get('S_t')}")
    L.append(f"- **S_t (measured, paired run):** {rep.get('S_t_measured')}")
    L.append(f"- **nc-Cu reached target:** {rep.get('nc_reached')}")
    L.append(f"- **Poly-Cu reached target:** {rep.get('poly_reached')}")
    L.append(f"- **ΔT (Q_gb):** +{rep.get('Delta_T_gb'):.1f} °C"
              if rep.get('Delta_T_gb') is not None else "- **ΔT (Q_gb):** N/A")
    if rep.get('Delta_T_95CI'):
        lo, hi = rep['Delta_T_95CI']
        L.append(f"- **ΔT 95% CI (Monte-Carlo):** [{lo:+.1f}, {hi:+.1f}] °C "
                  f"(median {rep.get('Delta_T_median'):+.1f})")
    L.append(f"- **S_P (pressure penalty):** {rep.get('S_P')}")
    L.append(f"- **Composite MCDA score Σ:** {rep.get('composite_score')}")
    L.append("")
    L.append("## Cost per joint (USD)")
    L.append("")
    L.append("| Line item | nc-Cu | Poly-Cu |")
    L.append("|---|---|---|")
    cn, cp = rep.get('cost_breakdown_nc') or {}, rep.get('cost_breakdown_poly') or {}
    for k in ('energy', 'machine', 'labor', 'au_material', 'cu_material', 'total'):
        L.append(f"| {k} | {cn.get(k, 0):.6f} | {cp.get(k, 0):.6f} |")
    L.append("")
    L.append("## Engineering quantities")
    L.append("")
    L.append(f"- Energy per joint — nc-Cu: {rep.get('energy_per_joint_nc_kWh')} kWh, "
              f"Poly-Cu: {rep.get('energy_per_joint_poly_kWh')} kWh")
    L.append(f"- CO₂ per joint — nc-Cu: {rep.get('co2_per_joint_nc_kg')} kg, "
              f"Poly-Cu: {rep.get('co2_per_joint_poly_kg')} kg")
    L.append(f"- Throughput — nc-Cu: {rep.get('throughput_nc_per_h')} j/h, "
              f"Poly-Cu: {rep.get('throughput_poly_per_h')} j/h")
    L.append(f"- Au utilisation — nc-Cu: {rep.get('eta_Au_nc')}, "
              f"Poly-Cu: {rep.get('eta_Au_poly')}")
    L.append(f"- Break-even Cu premium: {rep.get('break_even_premium_usd_kg')} USD/kg")
    L.append(f"- Break-even grain size: {rep.get('break_even_grain_size_nm')} nm")
    L.append("")
    L.append("## Narrative")
    L.append("")
    L.append(rep.get('narrative') or '_(no narrative provided)_')
    L.append("")
    return "\n".join(L)


def compute_sustainability_metrics(*, solver_history, params,
                                    target_c=0.06,
                                    d_current=None, d_ref=150.0,
                                    gb_width=1.5,
                                    Q_gb_J=100e3, Q_lat_J=200e3,
                                    T_bond_K=523.0,
                                    applied_stress_MPa=30.0,
                                    furnace_kW=2.0,
                                    grid_carbon_kg_per_kWh=0.40,
                                    batch_joints=100,
                                    energy_price_usd_kwh=0.10,
                                    machine_hourly_usd=50.0,
                                    cu_mass_kg=1e-3,
                                    au_mass_kg=None,
                                    nano_premium_usd_kg=50.0,
                                    sigma_Q_J=20e3,
                                    paired_results=None):
    """Comprehensive sustainability-metric computation.

    v3.4.16 changes (vs v3.4.15):
      * Bug #1 fix: all energy / CO₂ / cost lines now go through
        code_time_to_seconds() — picking time_unit='ns' no longer treats
        nanoseconds as seconds.
      * Bug #2 fix: _time_to_target returns (t, reached). The caller
        receives `nc_reached` and `poly_reached` flags; no silent fallback
        to t_arr[-1].
      * Bug #3 fix: ΔT uses effective_Q() (D-weighted average of Q_lat and
        Q_gb), not Q_gb alone. Q_eff_nc and Q_eff_poly are returned so the
        paper can state exactly which Q was used.
      * NEW: cost-per-joint breakdown for nc-Cu and Poly-Cu.
      * NEW: break-even Cu premium (USD/kg) and break-even grain size (nm).
      * NEW: composite MCDA score (geometric mean, scale-invariant).
      * NEW: Monte-Carlo 95% CI on ΔT from σ_Q.
      * NEW: optional `paired_results` argument — when run_paired_experiment()
        has been called, the measured S_t and reached flags override the
        geometric estimates.

    Parameters
    ----------
    solver_history : dict
        AuCuBondingSolver.history (must contain 'time' and 'c_bot').
    params : dict
        Simulation params. Reads D_lattice, D_gb, nc_grain_size_nm,
        gb_width_nm, applied_stress, time_unit.
    target_c : float
        Region-averaged Au concentration in the bottom layer required to
        consider the joint well-bonded (default 0.06).
    d_current, d_ref, gb_width : floats
        Grain sizes (nm) and GB width (nm) for the geometric S_t calc.
    Q_gb_J, Q_lat_J : floats
        Grain-boundary / lattice diffusion activation energies (J/mol).
        Used in the effective-Q and ΔT calculations.
    T_bond_K : float
        Bonding temperature in Kelvin.
    applied_stress_MPa : float
        Applied bonding pressure (MPa).
    furnace_kW, grid_carbon_kg_per_kWh, batch_joints, energy_price_usd_kwh,
    machine_hourly_usd, cu_mass_kg, au_mass_kg, nano_premium_usd_kg :
        Cost-model inputs (see compute_cost_per_joint + cost_per_joint
        in sustainability_figures.py).
    sigma_Q_J : float
        1-sigma literature uncertainty on Q_gb (default 20 kJ/mol).
    paired_results : dict or None
        If provided (output of run_paired_experiment), the measured
        t_reached / reached / S_t_measured override the geometric estimates.

    Returns
    -------
    dict — see build_sustainability_report() for the JSON-serialisable
    projection.
    """
    R = 8.314  # J/(mol·K)

    # ---- Effective grain sizes ----
    if d_current is None:
        d_current = float(params.get('nc_grain_size_nm', 95.0))
    d_current = max(float(d_current), 1.0)
    d_ref = max(float(d_ref), 1.0)
    gb_width = max(float(gb_width), 0.1)

    # ---- φ_gb and D_eff ----
    phi_gb_nc = gb_width / d_current
    phi_gb_poly = gb_width / d_ref
    D_lat = float(params.get('D_lattice', 0.01))
    D_gb = float(params.get('D_gb', 50.0))
    D_eff_nc = D_lat + (D_gb - D_lat) * phi_gb_nc
    D_eff_poly = D_lat + (D_gb - D_lat) * phi_gb_poly

    # ---- Q_eff (Bug #3 fix) ----
    Q_eff_nc_J = effective_Q(D_lat, D_gb, phi_gb_nc, Q_lat_J, Q_gb_J)
    Q_eff_poly_J = effective_Q(D_lat, D_gb, phi_gb_poly, Q_lat_J, Q_gb_J)

    # ---- Geometric S_t (always computable, even without a paired run) ----
    S_t = D_eff_nc / max(D_eff_poly, 1e-12)

    # ---- Time-to-target for nc-Cu (measured from solver.history) ----
    # Bug #2 fix: (t, reached) tuple — no silent NaN fallback.
    t_arr = _np_sus.asarray(solver_history.get('time', []), dtype=float)
    c_bot_arr = _np_sus.asarray(solver_history.get('c_bot', []), dtype=float)
    t_nc_measured, nc_reached = _time_to_target(t_arr, c_bot_arr, target_c)

    # ---- Override with paired-experiment measurements when available ----
    S_t_measured = float('nan')
    poly_reached = False
    if paired_results is not None:
        nc_run = paired_results.get('nc', {})
        poly_run = paired_results.get('poly', {})
        nc_reached = bool(nc_run.get('reached', nc_reached))
        poly_reached = bool(poly_run.get('reached', False))
        if nc_reached and 't_reached' in nc_run:
            t_nc_measured = float(nc_run['t_reached'])
        if poly_reached and 't_reached' in poly_run:
            t_poly_measured = float(poly_run['t_reached'])
            if t_nc_measured > 0 and _np_sus.isfinite(t_nc_measured):
                S_t_measured = t_poly_measured / t_nc_measured
                # Prefer the measured S_t for all downstream calculations.
                S_t = S_t_measured
        else:
            t_poly_measured = float('inf')
    else:
        # Geometric time-stretch fallback (the v3.4.15 behaviour, but now
        # ONLY when no paired run is supplied — and we surface the
        # `t_poly_estimated_is_geometric` flag so the UI can warn).
        if nc_reached:
            t_poly_measured = t_nc_measured * S_t
            poly_reached = False  # we didn't actually run Poly-Cu
        else:
            t_poly_measured = float(t_arr[-1] * S_t) if t_arr.size else float('inf')

    t_poly_estimated = t_poly_measured  # alias for backward compat

    # ---- ΔT from Arrhenius (Bug #3: uses Q_eff, not Q_gb alone) ----
    # We compute ΔT for the Poly-Cu side because that's where the
    # compensation is needed (Poly-Cu is the slower configuration).
    if S_t > 1.0 and Q_eff_poly_J > 0:
        # Exact Arrhenius inversion (more defensible than the linearisation):
        #   ΔT = [1/T - (R/Q)·ln(S_t)]^(-1) - T
        denom = 1.0 / T_bond_K - R * _np_sus.log(S_t) / Q_eff_poly_J
        if denom > 0:
            Delta_T_eff = (1.0 / denom) - T_bond_K
        else:
            Delta_T_eff = float('inf')
    else:
        Delta_T_eff = 0.0
    # Also keep the small-ΔT linearisations for backward compat / comparison
    if S_t > 1.0 and Q_gb_J > 0:
        Delta_T_gb = (R * T_bond_K**2 / Q_gb_J) * _np_sus.log(S_t)
    else:
        Delta_T_gb = 0.0
    if S_t > 1.0 and Q_lat_J > 0:
        Delta_T_lat = (R * T_bond_K**2 / Q_lat_J) * _np_sus.log(S_t)
    else:
        Delta_T_lat = 0.0

    # ---- S_P ≈ S_t (flux-balance approximation) ----
    S_P = S_t
    sigma_nc_MPa = float(applied_stress_MPa)
    sigma_poly_MPa = sigma_nc_MPa * S_P

    # ---- Au utilization ----
    eta_Au_nc = target_c
    c_bot_final = float(c_bot_arr[-1]) if c_bot_arr.size else 0.0
    eta_Au_poly = min(c_bot_final / S_t, target_c)

    # ---- Bug #1 fix: convert code-time to SI seconds for energy/CO₂ ----
    t_nc_s = code_time_to_seconds(t_nc_measured, params)
    t_poly_s = code_time_to_seconds(t_poly_measured, params)

    # ---- Energy, CO₂, throughput ----
    E_nc_kWh = furnace_kW * t_nc_s / 3600.0
    E_poly_kWh = furnace_kW * t_poly_s / 3600.0
    CO2_nc = E_nc_kWh * grid_carbon_kg_per_kWh
    CO2_poly = E_poly_kWh * grid_carbon_kg_per_kWh
    throughput_nc = 3600.0 / max(t_nc_s, 1e-12)
    throughput_poly = 3600.0 / max(t_poly_s, 1e-12)
    energy_ratio = E_poly_kWh / max(E_nc_kWh, 1e-12)
    co2_ratio = CO2_poly / max(CO2_nc, 1e-12)

    # ---- Cost per joint (energy + machine + material) ----
    # nc-Cu pays a nano-premium that decays exponentially with grain size
    # (smaller powders cost more). Poly-Cu pays zero premium.
    nano_prem_nc = nano_premium_usd_kg  # at d = d_current (user-controlled)
    cost_nc = compute_cost_per_joint(
        t_seconds=t_nc_s, furnace_kW=furnace_kW,
        batch_joints=batch_joints,
        energy_price_usd_kwh=energy_price_usd_kwh,
        machine_hourly_usd=machine_hourly_usd,
        au_mass_kg=au_mass_kg,
        cu_mass_kg=cu_mass_kg,
        bulk_cu_usd_kg=0.0,           # base price folded into premium below
        nc_cu_premium_usd_kg=nano_prem_nc,
    )
    cost_poly = compute_cost_per_joint(
        t_seconds=t_poly_s, furnace_kW=furnace_kW,
        batch_joints=batch_joints,
        energy_price_usd_kwh=energy_price_usd_kwh,
        machine_hourly_usd=machine_hourly_usd,
        au_mass_kg=au_mass_kg,
        cu_mass_kg=cu_mass_kg,
        bulk_cu_usd_kg=0.0,
        nc_cu_premium_usd_kg=0.0,      # Poly-Cu is bulk — no nano premium
    )
    cost_ratio = cost_poly['total'] / max(cost_nc['total'], 1e-12)

    # ---- Break-even Cu premium (USD/kg) ----
    # At what nano-Cu premium does the energy saving stop justifying the
    # material premium? = (cost_poly_excl_cu - cost_nc_excl_cu) / cu_mass
    if cu_mass_kg and cu_mass_kg > 0:
        poly_excl = cost_poly['total'] - cost_poly['cu_material']
        nc_excl = cost_nc['total'] - cost_nc['cu_material']
        break_even_premium = (poly_excl - nc_excl) / cu_mass_kg
    else:
        break_even_premium = float('nan')

    # ---- Break-even grain size (nm) ----
    # At what d_nc does the nc-Cu advantage drop below 1.5x?
    break_even_grain_size = d_ref / 1.5 if S_t > 1.5 else d_current

    # ---- Composite MCDA score ----
    composite = composite_sustainability_score(
        S_t=S_t, energy_ratio=energy_ratio, co2_ratio=co2_ratio,
        cost_ratio=cost_ratio)

    # ---- Monte-Carlo 95% CI on ΔT (only meaningful if Q_eff is finite) ----
    if S_t > 1.0 and _np_sus.isfinite(Delta_T_eff):
        mc = mc_sustainability_uncertainty(
            t_nc_s=t_nc_s, t_poly_s=t_poly_s,
            Q_gb_J=Q_eff_poly_J, Q_lat_J=Q_lat_J, T_K=T_bond_K,
            sigma_Q=sigma_Q_J)
        Delta_T_median = mc['Delta_T_median']
        Delta_T_95CI = mc['Delta_T_95CI']
    else:
        Delta_T_median = 0.0
        Delta_T_95CI = (0.0, 0.0)

    return {
        # Primary
        'S_t': float(S_t),
        'S_t_measured': float(S_t_measured),
        'nc_reached': bool(nc_reached),
        'poly_reached': bool(poly_reached),
        't_nc_measured': float(t_nc_measured),
        't_poly_measured': float(t_poly_measured),
        't_poly_estimated': float(t_poly_estimated),
        # Temperature
        'Delta_T_gb': float(Delta_T_gb),
        'Delta_T_lat': float(Delta_T_lat),
        'Delta_T_eff': float(Delta_T_eff),
        'Delta_T_median': float(Delta_T_median),
        'Delta_T_95CI': (float(Delta_T_95CI[0]), float(Delta_T_95CI[1])),
        # Pressure
        'S_P': float(S_P),
        'sigma_nc_MPa': float(sigma_nc_MPa),
        'sigma_poly_MPa': float(sigma_poly_MPa),
        # Geometry / physics
        'phi_gb_nc': float(phi_gb_nc),
        'phi_gb_poly': float(phi_gb_poly),
        'D_eff_nc': float(D_eff_nc),
        'D_eff_poly': float(D_eff_poly),
        'Q_eff_nc_kJmol': float(Q_eff_nc_J / 1e3),
        'Q_eff_poly_kJmol': float(Q_eff_poly_J / 1e3),
        # Au utilisation
        'eta_Au_nc': float(eta_Au_nc),
        'eta_Au_poly': float(eta_Au_poly),
        # Energy / CO₂ / throughput (Bug #1 fix: SI seconds)
        't_nc_s': float(t_nc_s),
        't_poly_s': float(t_poly_s),
        'E_nc_kWh': float(E_nc_kWh),
        'E_poly_kWh': float(E_poly_kWh),
        'CO2_nc': float(CO2_nc),
        'CO2_poly': float(CO2_poly),
        'throughput_nc': float(throughput_nc),
        'throughput_poly': float(throughput_poly),
        'energy_ratio': float(energy_ratio),
        'co2_ratio': float(co2_ratio),
        # Cost
        'cost_nc': cost_nc,
        'cost_poly': cost_poly,
        'cost_ratio': float(cost_ratio),
        'break_even_premium': float(break_even_premium),
        'break_even_grain_size': float(break_even_grain_size),
        # MCDA
        'composite_score': float(composite),
        # Target (so build_sustainability_report can find it)
        'target_c': float(target_c),
        # Constants
        'R': float(R),
    }


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Au-Cu Diffusion Bonding (Rectangular MPF)",
        layout="wide", initial_sidebar_state="expanded")

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
                '(Rectangular MPF) — v3.3</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#FFF8E7; padding:1rem; border-radius:8px;
                border-left:5px solid #C9A227;">
    <b>Physics:</b> MPF + Fickian/Hart + Solute Drag + Elastoplasticity (v2 core).<br>
    • Top <b>Poly-Cu</b> · Middle <b>Au film</b> · Bottom <b>nc-Cu</b>.<br>
    • 30 MPa downward → ∇σ_h drives Au <b>selectively into nc-Cu</b>.<br>
    • <b>Auto-substepping</b> + <b>mass-conserving rescale</b>.<br>
    <span class="viz-badge">v3.3: unified custom vmin/vmax across all views</span>
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
            nc_gs     = st.slider("nc-Cu grain size (nm)", 5.0, 150.0, 95.0, 1.0)

            # v3.4.15-sustainability: bottom-layer microstructure selector.
            # Switching this from "nc-Cu" to "Poly-Cu" forces the bottom Cu
            # layer to use the same grain size as the top Poly-Cu layer,
            # reducing φ_gb from ~0.10 to ~0.01 and D_eff by 10×. The
            # downstream physics (Hart diffusion, solute drag, spectral
            # elastoplasticity, GND tracking) all automatically reflect the
            # new φ_gb because it is computed from the order-parameter field.
            st.markdown("**🌱 Bottom layer microstructure (sustainability)")
            bottom_layer_type = st.selectbox(
                "Bottom layer type",
                ["nc-Cu (as-designed)",
                 "Poly-Cu (reference)",
                 "Coarse Poly-Cu (d=500 nm)"],
                key="bottom_layer_type",
                help=("Pick the bottom Cu layer microstructure. "
                      "\u00b7 nc-Cu uses the slider above (production design). "
                      "\u00b7 Poly-Cu (reference) forces the bottom layer to use "
                      "the SAME grain size as the top Poly-Cu (default 150 nm) — "
                      "this is the sustainability comparison baseline. "
                      "\u00b7 Coarse Poly-Cu (d=500 nm) replicates conventional "
                      "bulk / annealed Cu. Switching to Poly-Cu drops \u03c6_gb "
                      "from ~0.10 to ~0.01, reducing D_eff by 10\u00d7."))
            gb_w      = st.slider("GB width (nm)", 0.5, 5.0, 1.5, 0.1)
            curve_amp = st.slider("Poly-Cu boundary waviness (nm)",
                                   0.0, 20.0, 8.0, 1.0)
            use_warp = st.checkbox(
                "Apply warp to Voronoi boundaries", value=True)
            relax_st  = st.slider("Curvature relaxation steps", 0, 200, 10, 5,
                                   help="High values can absorb small grains.")
            seed_in   = st.number_input("Random seed", value=1234, step=1)

            st.subheader("🔲 Grid (rectangular)")
            # v3.4.7: User can now set the physical domain size Lx × Ly
            # directly (in nm). The code computes Nx = round(Lx/dx) and
            # Ny = round(Ly/dy) automatically and clamps to [64, 1024].
            # The original Nx/Ny sliders are kept as an "Advanced" expander
            # for users who want exact grid control.
            st.markdown("**📐 Physical domain size**")
            st.caption(
                "Set the physical width (Lx) and height (Ly) of the "
                "simulation domain in nm. The grid resolution (Nx, Ny) is "
                "computed automatically from dx, dy — keeping dx and dy "
                "fixed while you tune Lx and Ly. (Nx, Ny are clamped to "
                "[64, 1024].)")
            c1, c2 = st.columns(2)
            with c1:
                Lx_nm = st.slider(
                    "Lx (domain width, nm)", 50.0, 1600.0, 600.0, 10.0,
                    format="%.1f",
                    key="Lx_nm",
                    help="Physical width of the simulation domain in nm. "
                         "At dx=0.8 nm → Nx = round(Lx/dx). Example: Lx=600 nm "
                         "→ Nx=750.")
            with c2:
                Ly_nm = st.slider(
                    "Ly (domain height, nm)", 50.0, 1600.0, 300.0, 10.0,
                    format="%.1f",
                    key="Ly_nm",
                    help="Physical height of the simulation domain in nm. "
                         "At dy=0.8 nm → Ny = round(Ly/dy). Example: Ly=300 nm "
                         "→ Ny=375.")
            dx = st.slider("dx (nm)", 0.2, 2.0, 0.8, 0.1,
                            help="Grid spacing in x. Smaller dx = finer grid.")
            dy = st.slider("dy (nm)", 0.2, 2.0, 0.8, 0.1,
                            help="Grid spacing in y. Smaller dy = finer grid.")

            # Compute Nx, Ny from Lx, Ly, dx, dy (clamp to [64, 1024])
            Nx = int(round(Lx_nm / dx))
            Ny = int(round(Ly_nm / dy))
            Nx = max(64, min(1024, Nx))
            Ny = max(64, min(1024, Ny))
            # Recompute the *actual* Lx/Ly after rounding, so the caption
            # is honest (e.g. Lx=600, dx=0.8 → Nx=750 → Lx_actual=600.0).
            Lx_actual = Nx * dx
            Ly_actual = Ny * dy

            st.caption(
                f"📐 Actual domain: **{Lx_actual:.1f} nm × {Ly_actual:.1f} nm** "
                f"(Nx={Nx}, Ny={Ny}, {Nx*Ny:,} cells)")

            # Memory estimate (helps user avoid OOM)
            ncells = Nx * Ny
            mem_per_frame_mb = ncells * 8 / 1e6  # 8 bytes/float64 per field
            # With ~13 saved fields and the eta-list growth, multiply by ~50
            mem_estimate_mb = mem_per_frame_mb * 50
            if mem_estimate_mb > 1500:
                st.warning(
                    f"⚠️ Estimated memory ~{mem_estimate_mb/1000:.1f} GB — "
                    f"likely to OOM on Streamlit Cloud. Consider smaller "
                    f"Lx × Ly or larger dx.")
            elif mem_estimate_mb > 500:
                st.caption(
                    f"💾 Estimated memory ~{mem_estimate_mb:.0f} MB — OK "
                    f"but watch history length.")

            # Advanced: direct Nx/Ny override (rarely needed)
            with st.expander("🔧 Advanced: direct Nx/Ny override", expanded=False):
                st.caption(
                    "If you need an exact Nx or Ny that doesn't correspond "
                    "to round(Lx/dx), set it here. This will override the "
                    "Lx/Ly-driven values above.")
                adv_use_override = st.checkbox(
                    "Use direct Nx/Ny values from these sliders",
                    value=False, key="adv_use_override")
                Nx_adv = st.slider(
                    "Nx (direct)", 64, 1024, int(Nx), 4,
                    key="Nx_adv")
                Ny_adv = st.slider(
                    "Ny (direct)", 64, 1024, int(Ny), 4,
                    key="Ny_adv")
                if adv_use_override:
                    Nx, Ny = int(Nx_adv), int(Ny_adv)
                    Lx_actual = Nx * dx
                    Ly_actual = Ny * dy
                    st.caption(
                        f"↳ Override applied: Nx={Nx}, Ny={Ny} → "
                        f"{Lx_actual:.1f} nm × {Ly_actual:.1f} nm")

            # v3.4: time-unit selector — physics-preserving.
            st.markdown("**⏱️ Time unit (physics-preserving)**")
            st.caption(
                "Pick the unit you want to *display* time in. The numerical "
                "simulation only depends on the non-dimensional group "
                "`D·dt/dx²`, so relabelling the unit does NOT change the "
                "physics. What currently appears as `1.5 ns` total time can "
                "be displayed as `1.5 s` simply by selecting `s` — the dt "
                "slider and D slider keep the same numerical values (5e-3, "
                "50, …) but they are now interpreted as `5e-3 s` and "
                "`50 nm²/s` instead of `5e-3 ns` and `50 nm²/ns`. The "
                "diffusion pattern is IDENTICAL because D·dt is unchanged.")
            time_unit_keys = list(TIME_UNITS.keys())
            time_unit = st.selectbox(
                "Time unit",
                time_unit_keys,
                index=time_unit_keys.index(
                    st.session_state.get('time_unit_sel', DEFAULT_TIME_UNIT)),
                key="time_unit_sel",
                help="Pick the physical time unit for display. The simulation "
                     "is unchanged; only the labels on the time axis change.")

            # The dt slider uses the SAME range and default regardless of
            # the chosen unit. The unit interpretation is what changes.
            dt_label = f"dt ({time_unit})"
            dt = st.slider(dt_label, 1e-4, 5e-2, 5e-3, 1e-4, format="%.5f")
            st.caption(
                f"Domain: {Nx*dx:.1f} nm × {Ny*dy:.1f} nm  |  "
                f"dt = {dt:.5g} {time_unit}  →  total = "
                f"{dt * st.session_state.get('n_steps_cached', 300):.4g} "
                f"{time_unit}  (with current N-steps setting)")

            st.subheader("🌡️ Thermodynamics")
            W  = st.slider("W — chemical well depth", 0.1, 10.0, 2.0, 0.1)
            A  = st.slider("A — grain double-well",   0.1, 20.0, 5.0, 0.5)
            B  = st.slider("B — grain anti-overlap",  0.1, 30.0, 10.0, 0.5)
            kappa_eta = st.slider("κ_η (gradient)",  0.05, 5.0, 1.0, 0.05)
            aniso_str = st.slider("Anisotropy strength δ", 0.0, 0.8, 0.3, 0.05)

            st.subheader(f"🕳️ Diffusion (Hart) — D in nm²/{time_unit}")
            use_ch = st.checkbox(
                "Use Cahn-Hilliard for Au (uses W)", value=False,
                help="CH is more robust to stiff drift terms.")
            D_lat = st.slider(f"D_lattice (nm²/{time_unit})",
                                1e-4, 1.0, 0.01, 1e-3, format="%.4f")
            D_gb  = st.slider(f"D_gb (nm²/{time_unit}, grain boundaries)",
                                1.0, 100.0, 50.0, 1.0)
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
            st.session_state['n_steps_cached'] = int(n_steps)
            save_freq = st.slider("Save every N steps", 1, 100, 10, 1)

            # =========================================================
            # v3.4.8: AUTO-REBUILD GEOMETRY
            # =========================================================
            # Whenever ANY geometry-affecting parameter changes (Lx, Ly, dx,
            # dy, poly_frac, au_frac, n_poly_grains, poly_gs, nc_gs, gb_w,
            # curve_amp, use_warp, relax_st, seed_in), we rebuild the cached
            # geometry (gm0, c0, eta0, meta0) automatically — no Initialize
            # button needed. The Initialize button is kept for explicit
            # re-initialization when desired (e.g. after changing non-
            # geometry params).
            #
            # We track the previous values of these geometry-affecting
            # params in session_state; if any differ from the cached ones,
            # we trigger a rebuild inline.
            GEOM_TRIGGER_KEYS = [
                ('Lx_nm',         Lx_nm),
                ('Ly_nm',         Ly_nm),
                ('dx',            dx),
                ('dy',            dy),
                ('poly_frac',     poly_frac),
                ('au_frac',       au_frac),
                ('n_poly_grains', int(n_poly_grains)),
                ('poly_gs',       poly_gs),
                ('nc_gs',         nc_gs),
                ('gb_w',          gb_w),
                ('curve_amp',     curve_amp),
                ('use_warp',      bool(use_warp)),
                ('relax_st',      relax_st),
                ('seed_in',       int(seed_in)),
                ('bottom_layer_type', bottom_layer_type),
            ]

            # Helper that builds params dict, validates, and rebuilds the
            # cached geometry. Returns True on success.
            def _do_initialize(silent=False):
                # v3.4.15-sustainability: pick the effective bottom-layer grain
                # size based on the sidebar toggle. The "as-designed" nc-Cu
                # path uses the user's nc_gs slider value; the Poly-Cu
                # reference forces the bottom layer to use the SAME grain size
                # as the top Poly-Cu layer (so φ_gb drops from ~0.10 to ~0.01);
                # the coarse Poly-Cu option uses a fixed 500 nm grain size,
                # replicating conventional bulk / annealed Cu.
                if bottom_layer_type == "Poly-Cu (reference)":
                    nc_gs_effective = float(poly_gs)
                elif bottom_layer_type == "Coarse Poly-Cu (d=500 nm)":
                    nc_gs_effective = 500.0
                else:
                    nc_gs_effective = float(nc_gs)
                params = {
                    'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy, 'dt': dt,
                    'time_unit': time_unit,
                    'poly_thickness_frac': poly_frac,
                    'au_thickness_frac':   au_frac,
                    'n_poly_grains':       int(n_poly_grains),
                    'poly_grain_size_nm':  poly_gs,
                    'nc_grain_size_nm':    nc_gs_effective,
                    'bottom_layer_type':   bottom_layer_type,
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
                    return False
                for w in warns:
                    st.warning(w)
                st.session_state.params = params
                st.session_state.initialized = True
                geom = AuCuBondingGeometry(Nx, Ny, dx, dy)
                res = geom.build(
                    poly_thickness_frac=poly_frac,
                    au_thickness_frac=au_frac,
                    n_poly_grains=int(n_poly_grains),
                    poly_grain_size_nm=poly_gs,
                    nc_grain_size_nm=nc_gs_effective,
                    gb_width_nm=gb_w,
                    poly_curve_amp=curve_amp,
                    seed=int(seed_in),
                    use_warp=bool(use_warp))
                if res is FAILED or res is None:
                    st.error(f"Geometry build failed: {FAILED.reason}")
                    return False
                c, eta_list, gm, lid, meta = res
                st.session_state.geom = geom
                st.session_state.c0 = c
                st.session_state.eta0 = eta_list
                st.session_state.gm0 = gm
                st.session_state.meta0 = meta
                # Also clear any stale solver history so the user doesn't
                # see results from an OLD domain size.
                for _k in ('history', 'timesteps', 'solver'):
                    if _k in st.session_state:
                        del st.session_state[_k]
                if not silent:
                    st.success(
                        f"✅ Initialized: {meta['n_poly']} Poly grain(s) "
                        f"+ 1 Au + {meta['n_nc']} nc = {meta['n_total']} phases "
                        f"(domain {Nx*dx:.1f} × {Ny*dy:.1f} nm)")
                return True

            # Detect any geometry-affecting param change
            prev_geom_params = st.session_state.get('_prev_geom_params', {})
            geom_changed = False
            for k, v in GEOM_TRIGGER_KEYS:
                if prev_geom_params.get(k) != v:
                    geom_changed = True
                    prev_geom_params[k] = v
            st.session_state['_prev_geom_params'] = prev_geom_params

            # Auto-rebuild on geometry change (first run + every slider drag)
            if geom_changed and st.session_state.get('initialized', False):
                # Silent rebuild — don't show a success toast every time
                # the user nudges Lx by 10 nm. Use a spinner so they see
                # what's happening.
                with st.spinner("🔄 Rebuilding geometry…"):
                    _do_initialize(silent=True)
                # Force a rerun so all downstream tabs see the new geometry
                st.rerun()

            # Explicit Initialize button (kept for parity with v3.4.7)
            if st.button("🚀 Initialize", type="primary",
                         use_container_width=True):
                _do_initialize(silent=False)

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
                        "📤 Export", "🌱 Sustainability"])

        # ---- Geometry ----------------------------------------------
        with tabs[0]:
            st.header("Initial 3-Layer Geometry (Rectangular)")
            st.caption(
                f"Poly-Cu: {meta['n_poly']} coarse grain(s) · "
                f"Au: 1 flat slab · nc-Cu: {meta['n_nc']} fine grains · "
                f"Total = {meta['n_total']} phases")
            st.caption(
                f"Domain: {Nx*dx:.1f} nm × {Ny*dy:.1f} nm  |  "
                f"grid: {Nx} × {Ny}")

            extent = [-Nx*dx/2, Nx*dx/2, -Ny*dy/2, Ny*dy/2]
            x_coords = np.linspace(extent[0], extent[1], Nx)
            gm = st.session_state.gm0

            fig, ax = plt.subplots(figsize=(10, 4.5))
            cmap = ColormapRegistry.resolve('grain_map')
            im = ax.imshow(gm, extent=extent, origin='lower', cmap=cmap,
                           interpolation='nearest', aspect='equal')

            _plot_interface(ax, x_coords, meta['y_poly_bot_eff'],
                            color='white', lw=1.5, ls='--',
                            label='Poly-Cu / Au interface')
            _plot_interface(ax, x_coords, meta['y_au_bot_eff'],
                            color='yellow', lw=2.0,
                            label='Au / nc-Cu interface')

            ax.set_title(
                f"Voronoi grain map — Poly-Cu: {meta['n_poly']} grain(s), "
                f"nc-Cu: {meta['n_nc']} grains")
            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
            ax.legend(loc='upper right', framealpha=0.85)
            st.pyplot(fig); plt.close(fig)

            c0 = st.session_state.c0
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fs_c = viz_cfg['field_settings'].get('c', {})
            cmap_c = ColormapRegistry.resolve(
                FIELD_DEFAULT_CMAPS['c'],
                invert=fs_c.get('invert', False))
            im = ax.imshow(c0, extent=extent, origin='lower', cmap=cmap_c,
                           vmin=0, vmax=1, aspect='equal')
            _plot_interface(ax, x_coords, meta['y_poly_bot_eff'],
                            color='white', lw=1.5, ls='--')
            _plot_interface(ax, x_coords, meta['y_au_bot_eff'],
                            color='yellow', lw=2.0)
            plt.colorbar(im, ax=ax, label='c (Au)')
            ax.set_title(
                f"Initial Au concentration · film thickness ≈ "
                f"{meta['y_poly_bot']-meta['y_au_bot']:.1f} nm")
            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
            st.pyplot(fig); plt.close(fig)

            with st.expander("Individual Poly-Cu η fields", expanded=False):
                eta0 = st.session_state.eta0
                n_show = min(int(meta['n_poly']), 5)
                if n_show > 0:
                    fig, axes = plt.subplots(1, n_show,
                                              figsize=(4.5 * n_show, 4.0))
                    axes = np.atleast_1d(axes)
                    for i in range(n_show):
                        axes[i].imshow(eta0[i], extent=extent, origin='lower',
                                       cmap='Blues', vmin=0, vmax=1,
                                       aspect='equal')
                        axes[i].set_title(f"η_{i} (Poly grain {i+1})")
                        axes[i].set_xlabel('x (nm)')
                        axes[i].set_ylabel('y (nm)')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)

        # ---- Run ---------------------------------------------------
        with tabs[1]:
            st.header("Run the MPF Simulation")
            col1, col2 = st.columns([2, 1])
            with col1:
                ch_status = ('ON (uses W)' if params.get('use_cahn_hilliard')
                             else 'OFF (Fickian)')
                t_unit = get_time_unit_label(params)
                st.markdown(
                    f"**Expected physics with current settings:**\n"
                    f"- D_lattice = `{params['D_lattice']:.4f}` nm²/{t_unit}, "
                    f"D_gb = `{params['D_gb']:.1f}` nm²/{t_unit}\n"
                    f"- Ω (stress drift) = `{params['Omega']:.3f}` "
                    f"(σ_h normalized by 1 GPa)\n"
                    f"- Cahn-Hilliard mode = `{ch_status}`\n"
                    f"- Applied stress = `{params['applied_stress']/1e6:.0f} MPa @ `"
                    f"`{params['applied_stress_angle']:.0f}°`\n"
                    f"- Au-Cu misfit β = `{params['misfit_beta']:.3f}`\n"
                    f"- Solute drag α = `{params['alpha_drag']:.0f}`\n"
                    f"- Poly-Cu grains = `{params['n_poly_grains']}`\n"
                    f"- Steps × dt = `{params['n_steps']} × "
                    f"{params['dt']:.4g} {t_unit}`"
                    f" = `{params['n_steps']*params['dt']:.3g} {t_unit}` total"
                    f"\n- Non-dim. group `D_gb·dt/dx²` (Fourier) = "
                    f"`{params['D_gb']*params['dt']/max(params['dx']**2, 1e-9):.4g}`"
                    f" — physics is preserved across any unit choice.")
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
                        if res is FAILED or res is None:
                            st.error(f"Solver step failed: {FAILED.reason}")
                            break
                        if step % save_freq == 0 or step == n_steps - 1:
                            history.append({
                                k: (v.copy() if isinstance(v, np.ndarray)
                                    else (dict(v) if isinstance(v, dict) else v))
                                for k, v in res.items()
                            })
                            timesteps.append(step * params['dt'])
                        progress.progress((step + 1) / n_steps)
                        if step % max(1, n_steps // 20) == 0:
                            mass_ratio = (solver.history['au_mass'][-1] /
                                          max(solver.history['au_mass'][0], 1e-12))
                            status.text(
                                f"Step {step+1}/{n_steps} · "
                                f"c_top={solver.history['c_top'][-1]:.4f} · "
                                f"c_bot={solver.history['c_bot'][-1]:.4f} · "
                                f"Δc="
                                f"{solver.history['c_bot'][-1]-solver.history['c_top'][-1]:+.4f} · "
                                f"mass/init={mass_ratio:.4f}")

                    st.success(f"✅ Done. {len(history)} frames saved.")
                    st.session_state.history = history
                    st.session_state.timesteps = timesteps
                    st.session_state.solver = solver
                    # Save the scalar diagnostics (1D time-series) alongside
                    # the 2D field history. This bridges the saved temporal
                    # data to the Compare Saved (Temporal) and Sustainability
                    # tabs, which now compute measured S_t from actual saved
                    # runs instead of relying on the active in-memory solver.
                    sid = SimulationDatabase.save(
                        params, history, None,
                        run_time=n_steps * params['dt'],
                        diagnostics=solver.history)
                    st.session_state.last_sid = sid
                    st.balloons()

        # ---- Visualization Settings --------------------------------
        with tabs[2]:
            st.header("🎨 Visualization Settings")
            st.caption("These settings apply to the Results, Line Profiles, "
                       "Plotly, and Export tabs.")

            sub1, sub2, sub3 = st.tabs(
                ["Global & Field Scaling",
                 "Overlay & Vector & Animation",
                 "Figure Customization (v3.4)"])

            with sub1:
                st.subheader("🖌️ Global matplotlib theme")
                theme_keys = list(MATPLOTLIB_THEMES.keys())
                default_theme_idx = (theme_keys.index(viz_cfg['theme'])
                                     if viz_cfg['theme'] in theme_keys else 0)
                c1, c2, c3 = st.columns(3)
                with c1:
                    theme = st.selectbox(
                        "Theme", theme_keys,
                        index=default_theme_idx,
                        key="viz_theme")
                with c2:
                    title_fs = st.slider("Title font size",
                                          6, 24, int(viz_cfg['title_font_size']),
                                          key="viz_title_fs")
                    label_fs = st.slider("Axis label font size",
                                          6, 20, int(viz_cfg['label_font_size']),
                                          key="viz_label_fs")
                with c3:
                    tick_fs = st.slider("Tick font size",
                                        4, 18, int(viz_cfg['tick_font_size']),
                                        key="viz_tick_fs")
                    legend_fs = st.slider("Legend font size",
                                          4, 18, int(viz_cfg['legend_font_size']),
                                          key="viz_legend_fs")
                dpi = st.slider("DPI (matplotlib figures & exports)",
                                72, 600, int(viz_cfg['dpi']), 12,
                                key="viz_dpi")
                cb1, cb2 = st.columns(2)
                with cb1:
                    cb_frac = st.slider("Colorbar fraction",
                                         0.02, 0.15,
                                         float(viz_cfg['colorbar_fraction']),
                                         0.005, key="viz_cb_frac")
                with cb2:
                    cb_pad = st.slider("Colorbar pad",
                                        0.01, 0.15,
                                        float(viz_cfg['colorbar_pad']),
                                        0.005, key="viz_cb_pad")

                if st.button("♻️ Apply theme now", key="viz_apply_theme"):
                    viz_cfg['theme'] = theme
                    viz_cfg['title_font_size'] = title_fs
                    viz_cfg['label_font_size'] = label_fs
                    viz_cfg['tick_font_size'] = tick_fs
                    viz_cfg['legend_font_size'] = legend_fs
                    viz_cfg['dpi'] = dpi
                    viz_cfg['colorbar_fraction'] = cb_frac
                    viz_cfg['colorbar_pad'] = cb_pad
                    apply_matplotlib_theme(theme, viz_cfg)
                    st.success("Theme applied.")
                    st.rerun()

                st.markdown("---")
                st.subheader("📏 Per-field dynamic range & scaling")
                field_keys = list(FIELD_DEFAULT_CMAPS.keys())
                field_choice = st.selectbox(
                    "Field to configure", field_keys,
                    key="viz_field_choice")
                fs = viz_cfg['field_settings'][field_choice]

                auto = st.checkbox("Auto range (2–98 percentile)",
                                    value=fs.get('auto', True),
                                    key=f"viz_auto_{field_choice}")
                if not auto:
                    c1, c2 = st.columns(2)
                    # v3.3: seed the inputs from FIELD_DEFAULT_RNG when empty
                    drng = FIELD_DEFAULT_RNG.get(field_choice, (0.0, 1.0))
                    current_vmin = (fs.get('vmin')
                                    if fs.get('vmin') is not None
                                    else drng[0])
                    current_vmax = (fs.get('vmax')
                                    if fs.get('vmax') is not None
                                    else drng[1])
                    vmin_in = c1.number_input(
                        "Min value (vmin)",
                        value=float(current_vmin),
                        key=f"viz_vmin_{field_choice}",
                        format="%.6g")
                    vmax_in = c2.number_input(
                        "Max value (vmax)",
                        value=float(current_vmax),
                        key=f"viz_vmax_{field_choice}",
                        format="%.6g")
                else:
                    vmin_in, vmax_in = None, None

                col_log, col_inv = st.columns(2)
                with col_log:
                    log_scale = st.checkbox(
                        "Logarithmic scale",
                        value=fs.get('log', False),
                        key=f"viz_log_{field_choice}",
                        help="Best for D_eff, rho_gnd; values ≤ 0 are clipped.")
                with col_inv:
                    invert = st.checkbox(
                        "Invert colormap",
                        value=fs.get('invert', False),
                        key=f"viz_inv_{field_choice}")

                # v3.4: per-field colormap picker (80+ options)
                cmap_options = ColormapRegistry.list_all()
                current_cmap = fs.get('cmap',
                                       FIELD_DEFAULT_CMAPS.get(field_choice, 'viridis'))
                cmap_idx = (cmap_options.index(current_cmap)
                            if current_cmap in cmap_options else 0)
                cmap_choice = st.selectbox(
                    f"Colormap for '{field_choice}'",
                    cmap_options,
                    index=cmap_idx,
                    key=f"viz_cmap_{field_choice}",
                    help=f"Pick from {len(cmap_options)} colormaps. "
                         f"This is used by multi-field grid, overlay, "
                         f"vector overlay, Plotly, animation, export.")

                # Live preview swatch
                try:
                    swatch_cmap = ColormapRegistry.resolve(cmap_choice,
                                                            invert=invert)
                    gradient = np.linspace(0, 1, 256).reshape(1, -1)
                    fig_sw, ax_sw = plt.subplots(figsize=(4.2, 0.45),
                                                  dpi=80)
                    ax_sw.imshow(gradient, aspect='auto', cmap=swatch_cmap)
                    ax_sw.set_axis_off()
                    st.pyplot(fig_sw, clear_figure=True); plt.close(fig_sw)
                except Exception:
                    pass

                if st.button("💾 Save field settings",
                              key=f"viz_save_{field_choice}"):
                    viz_cfg['field_settings'][field_choice] = {
                        'auto': auto,
                        'vmin': vmin_in,
                        'vmax': vmax_in,
                        'log': log_scale,
                        'invert': invert,
                        'cmap': cmap_choice,
                    }
                    st.success(f"Saved settings for '{field_choice}' "
                                f"(cmap = '{cmap_choice}').")
                    st.rerun()

                if 'history' in st.session_state and st.session_state.history:
                    idx_prev = st.slider("Preview frame", 0,
                                          len(st.session_state.history) - 1,
                                          len(st.session_state.history) - 1,
                                          key="viz_prev_frame")
                    d_prev, unit_prev = _rescale_field(
                        field_choice,
                        st.session_state.history[idx_prev][field_choice])
                    fig_cfg_prev = viz_cfg.get('figure_customization', {})
                    fig_w_prev = float(fig_cfg_prev.get('overlay_width', 8.0))
                    fig_h_prev = float(fig_cfg_prev.get('overlay_height', 4.0))
                    fig, ax = plt.subplots(figsize=(fig_w_prev, fig_h_prev))
                    fs_prev = viz_cfg['field_settings'][field_choice]
                    cmap_name_prev = fs_prev.get('cmap',
                                                  FIELD_DEFAULT_CMAPS.get(field_choice, 'viridis'))
                    cmap_prev = ColormapRegistry.resolve(
                        cmap_name_prev,
                        invert=fs_prev.get('invert', False))
                    prev_extent = [-Nx*dx/2, Nx*dx/2, -Ny*dy/2, Ny*dy/2]
                    if field_choice in DISCRETE_FIELDS:
                        ax.imshow(d_prev, extent=prev_extent,
                                  origin='lower', cmap=cmap_prev,
                                  interpolation='nearest', aspect='equal')
                    else:
                        norm, vmin, vmax = compute_field_norm(
                            field_choice, d_prev, fs_prev,
                            default_rng=FIELD_DEFAULT_RNG.get(field_choice))
                        if norm is not None:
                            im = ax.imshow(d_prev, extent=prev_extent,
                                            origin='lower', cmap=cmap_prev,
                                            norm=norm, interpolation='bilinear',
                                            aspect='equal')
                        else:
                            im = ax.imshow(d_prev, extent=prev_extent,
                                            origin='lower', cmap=cmap_prev,
                                            vmin=vmin, vmax=vmax,
                                            interpolation='bilinear',
                                            aspect='equal')
                        if fig_cfg_prev.get('show_colorbar', True):
                            cbar_orient = fig_cfg_prev.get('colorbar_orientation',
                                                             'vertical')
                            plt.colorbar(im, ax=ax, orientation=cbar_orient,
                                         label=f"{field_choice} ({unit_prev})")
                    ax.set_title(f"Preview: {field_choice}  (cmap = '{cmap_name_prev}')")
                    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
                    apply_axis_styling(ax, fig_cfg_prev)
                    st.pyplot(fig); plt.close(fig)

            with sub2:
                st.subheader("🪟 Multi-field overlay builder")
                ov = viz_cfg['overlay']
                ov_enabled = st.checkbox("Enable overlay mode in Results",
                                          value=ov['enabled'], key="ov_en")
                field_keys2 = list(FIELD_DEFAULT_CMAPS.keys())
                c1, c2 = st.columns(2)
                with c1:
                    base_idx = (field_keys2.index(ov['base'])
                                if ov['base'] in field_keys2 else 7)
                    ov_base = st.selectbox("Base field", field_keys2,
                                            index=base_idx, key="ov_base")
                with c2:
                    over_idx = (field_keys2.index(ov['overlay'])
                                if ov['overlay'] in field_keys2 else 5)
                    ov_over = st.selectbox("Overlay field", field_keys2,
                                            index=over_idx, key="ov_over")
                c1, c2 = st.columns(2)
                with c1:
                    ov_mode = st.radio("Overlay mode",
                                        ['contour', 'heatmap'],
                                        index=0 if ov['mode'] == 'contour' else 1,
                                        key="ov_mode")
                with c2:
                    ov_alpha = st.slider("Alpha", 0.05, 1.0,
                                          float(ov['alpha']), 0.05,
                                          key="ov_alpha")
                ov_nc = st.slider("Contour levels", 3, 30,
                                   int(ov['n_contours']), 1, key="ov_nc")
                ov_ghost = st.checkbox("Ghost grain-map underneath",
                                        value=ov['show_grain_ghost'],
                                        key="ov_ghost")
                if st.button("💾 Save overlay settings", key="ov_save"):
                    viz_cfg['overlay'] = {
                        'enabled': ov_enabled, 'base': ov_base,
                        'overlay': ov_over, 'alpha': ov_alpha,
                        'mode': ov_mode, 'n_contours': ov_nc,
                        'show_grain_ghost': ov_ghost,
                    }
                    st.success("Overlay settings saved.")
                    st.rerun()

                st.markdown("---")
                st.subheader("🧭 Vector field overlay")
                vv = viz_cfg['vector']
                vv_enabled = st.checkbox("Enable vector-field mode",
                                          value=vv['enabled'], key="vv_en")
                c1, c2 = st.columns(2)
                with c1:
                    vv_src_idx = (VECTOR_FIELD_SOURCES.index(vv['source'])
                                  if vv['source'] in VECTOR_FIELD_SOURCES else 0)
                    vv_source = st.selectbox("Source field",
                                              VECTOR_FIELD_SOURCES,
                                              index=vv_src_idx,
                                              key="vv_src")
                with c2:
                    vv_mode_list = ['gradient', 'flux', 'principal_stress']
                    vv_m_idx = (vv_mode_list.index(vv['mode'])
                                if vv['mode'] in vv_mode_list else 0)
                    vv_mode = st.selectbox("Mode", vv_mode_list,
                                            index=vv_m_idx,
                                            key="vv_mode")
                c1, c2 = st.columns(2)
                with c1:
                    vv_density = st.slider("Arrow density", 6, 80,
                                            int(vv['density']), 2, key="vv_den")
                with c2:
                    vv_scale = st.slider("Arrow scale", 0.1, 5.0,
                                          float(vv['scale']), 0.1, key="vv_scale")
                vv_color = st.color_picker("Arrow color",
                                            value='#ffffff' if vv['color'] == 'white' else vv['color'],
                                            key="vv_col")
                if st.button("💾 Save vector settings", key="vv_save"):
                    viz_cfg['vector'] = {
                        'enabled': vv_enabled, 'source': vv_source,
                        'mode': vv_mode, 'density': vv_density,
                        'color': vv_color, 'scale': vv_scale,
                    }
                    st.success("Vector settings saved.")
                    st.rerun()

                st.markdown("---")
                st.subheader("🎬 Animation settings")
                an = viz_cfg['animation']
                c1, c2, c3 = st.columns(3)
                with c1:
                    an_fmt = st.selectbox("Format", ['gif', 'mp4'],
                                           index=0 if an['format'] == 'gif' else 1,
                                           key="an_fmt")
                with c2:
                    an_fps = st.slider("FPS", 1, 30, int(an['fps']), 1,
                                        key="an_fps")
                with c3:
                    an_dpi = st.slider("DPI", 72, 400, int(an['dpi']), 12,
                                        key="an_dpi")
                an_wm = st.checkbox("Show watermark",
                                     value=an['watermark'], key="an_wm")
                an_wm_text = st.text_input("Watermark text",
                                            value=an['watermark_text'],
                                            key="an_wm_text")
                wm_pos_list = ['lower right', 'lower left', 'upper right', 'upper left']
                wm_idx = (wm_pos_list.index(an['watermark_pos'])
                          if an['watermark_pos'] in wm_pos_list else 0)
                an_wm_pos = st.selectbox("Watermark position",
                                          wm_pos_list,
                                          index=wm_idx,
                                          key="an_wm_pos")
                if st.button("💾 Save animation settings", key="an_save"):
                    viz_cfg['animation'] = {
                        'format': an_fmt, 'fps': an_fps, 'dpi': an_dpi,
                        'watermark': an_wm, 'watermark_text': an_wm_text,
                        'watermark_pos': an_wm_pos,
                    }
                    st.success("Animation settings saved.")
                    st.rerun()

            # =========================================================
            # v3.4: PER-FIGURE CUSTOMIZATION
            # =========================================================
            with sub3:
                fc = viz_cfg['figure_customization']
                st.info(
                    "💡 **v3.4.6**: You can now customize each figure inline "
                    "by opening the 🎨 expander panel that appears above the "
                    "figure in the Results, Line Profiles, Plotly, and Export "
                    "tabs. Those inline panels update the figure immediately "
                    "with no Save button needed. The settings below are the "
                    "global defaults and still work — but you must click "
                    "💾 Save figure customization to apply them.")
                st.caption(
                    "Global defaults applied to multi-field grid, overlay, "
                    "vector overlay, diagnostics, line-profile (matplotlib) "
                    "and 3D surface (Plotly) figures. Leave text overrides "
                    "blank to use defaults. Set xmin/xmax/ymin/ymax to `0` "
                    "for auto (or use the toggle).")

                st.markdown("#### 📐 Figure sizes (inches)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption("**Multi-field grid**")
                    _mf_w = float(max(4.0, min(30.0, fc.get('multi_field_width', 13.5))))
                    mf_w = st.number_input("Width",  value=_mf_w,
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_mf_w")
                    _mf_h = float(max(3.0, min(30.0, fc.get('multi_field_height', 11.0))))
                    mf_h = st.number_input("Height", value=_mf_h,
                                            min_value=3.0, max_value=30.0, step=0.5,
                                            key="fc_mf_h")
                with c2:
                    st.caption("**Overlay / Vector**")
                    _ov_w = float(max(4.0, min(30.0, fc.get('overlay_width', 11.0))))
                    ov_w = st.number_input("Width",  value=_ov_w,
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_ov_w")
                    _ov_h = float(max(3.0, min(30.0, fc.get('overlay_height', 5.5))))
                    ov_h = st.number_input("Height", value=_ov_h,
                                            min_value=3.0, max_value=30.0, step=0.5,
                                            key="fc_ov_h")
                with c3:
                    st.caption("**Diagnostics**")
                    # v3.4.11: clamp value to [min, max] before passing to
                    # st.number_input to avoid StreamlitValueBelowMinError /
                    # StreamlitValueAboveMaxError when the value was set by
                    # a different widget (e.g. the quick size control above
                    # the diagnostics figure, or the enhanced panel).
                    _dg_w_raw = fc.get('diagnostics_width', 13.0)
                    _dg_w = float(max(4.0, min(30.0, _dg_w_raw)))
                    dg_w = st.number_input("Width",  value=_dg_w,
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_dg_w")
                    _dg_h_raw = fc.get('diagnostics_height', 8.0)
                    _dg_h = float(max(3.0, min(30.0, _dg_h_raw)))
                    dg_h = st.number_input("Height", value=_dg_h,
                                            min_value=3.0, max_value=30.0, step=0.5,
                                            key="fc_dg_h")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("**Line profile**")
                    _lp_w = float(max(4.0, min(30.0, fc.get('line_profile_width', 11.0))))
                    lp_w = st.number_input("Width",  value=_lp_w,
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_lp_w")
                    _lp_h = float(max(2.0, min(20.0, fc.get('line_profile_height', 4.5))))
                    lp_h = st.number_input("Height", value=_lp_h,
                                            min_value=2.0, max_value=20.0, step=0.5,
                                            key="fc_lp_h")
                with c2:
                    st.caption("**Vector overlay**")
                    _vc_w = float(max(4.0, min(30.0, fc.get('vector_width', 11.0))))
                    vc_w = st.number_input("Width",  value=_vc_w,
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_vc_w")
                    _vc_h = float(max(3.0, min(20.0, fc.get('vector_height', 5.5))))
                    vc_h = st.number_input("Height", value=_vc_h,
                                            min_value=3.0, max_value=20.0, step=0.5,
                                            key="fc_vc_h")

                st.markdown("---")
                st.markdown("#### ✏️ Line / marker styling")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    lw_main = st.number_input("Line width (main)",  value=float(fc.get('line_width', 2.0)),
                                                min_value=0.2, max_value=10.0, step=0.1,
                                                key="fc_lw")
                with c2:
                    lw_minor = st.number_input("Line width (minor)",  value=float(fc.get('line_width_minor', 1.5)),
                                                min_value=0.1, max_value=8.0, step=0.1,
                                                key="fc_lwm")
                with c3:
                    # v3.4.1: all int (StreamlitMixedNumericTypesError fix)
                    _ms_val = fc.get('marker_size', 6)
                    try:
                        _ms_val = int(_ms_val) if _ms_val is not None else 6
                    except (TypeError, ValueError):
                        _ms_val = 6
                    ms = st.number_input("Marker size", value=_ms_val,
                                          min_value=1, max_value=30, step=1,
                                          key="fc_ms")
                with c4:
                    mw = st.number_input("Marker width", value=float(fc.get('marker_width', 1.0)),
                                          min_value=0.1, max_value=5.0, step=0.1,
                                          key="fc_mw")
                c1, c2, c3 = st.columns(3)
                with c1:
                    cw = st.number_input("Contour line width",
                                          value=float(fc.get('contour_line_width', 1.0)),
                                          min_value=0.1, max_value=5.0, step=0.1,
                                          key="fc_cw")
                with c2:
                    qw = st.number_input("Quiver width",
                                          value=float(fc.get('quiver_width', 0.0025)),
                                          min_value=0.0001, max_value=0.02, step=0.0005,
                                          format="%.5f",
                                          key="fc_qw")
                with c3:
                    qs = st.number_input("Quiver scale",
                                          value=float(fc.get('quiver_scale', 30.0)),
                                          min_value=1.0, max_value=200.0, step=1.0,
                                          key="fc_qs")

                st.markdown("---")
                st.markdown("#### 🏷️ Text overrides (leave blank = default)")
                ti_ov = st.text_input("Title override", value=fc.get('title_override', ''),
                                       key="fc_ti", help="Replaces every figure's title.")
                xl_ov = st.text_input("X-axis label override",
                                       value=fc.get('xlabel_override', ''),
                                       key="fc_xl")
                yl_ov = st.text_input("Y-axis label override",
                                       value=fc.get('ylabel_override', ''),
                                       key="fc_yl")

                st.markdown("---")
                st.markdown("#### 📊 Grid & axis limits")
                c1, c2 = st.columns(2)
                with c1:
                    show_grid = st.checkbox("Show grid", value=bool(fc.get('show_grid', True)),
                                              key="fc_grid")
                    grid_alpha = st.slider("Grid alpha", 0.0, 1.0,
                                            float(fc.get('grid_alpha', 0.3)),
                                            0.05, key="fc_grid_a")
                with c2:
                    axis_equal = st.checkbox("Aspect ratio = 'equal'",
                                              value=bool(fc.get('axis_equal', True)),
                                              key="fc_eq")
                    tight_layout = st.checkbox("Use tight_layout",
                                                value=bool(fc.get('tight_layout', True)),
                                                key="fc_tight_layout")

                xlim_cur = fc.get('xlim', (None, None))
                ylim_cur = fc.get('ylim', (None, None))
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    xmin_in = st.text_input("xmin (blank=auto)",
                                             value='' if xlim_cur[0] is None else str(xlim_cur[0]),
                                             key="fc_xmin")
                with c2:
                    xmax_in = st.text_input("xmax (blank=auto)",
                                             value='' if xlim_cur[1] is None else str(xlim_cur[1]),
                                             key="fc_xmax")
                with c3:
                    ymin_in = st.text_input("ymin (blank=auto)",
                                             value='' if ylim_cur[0] is None else str(ymin_cur[0]),
                                             key="fc_ymin")
                with c4:
                    ymax_in = st.text_input("ymax (blank=auto)",
                                             value='' if ylim_cur[1] is None else str(ylim_cur[1]),
                                             key="fc_ymax")

                st.markdown("---")
                st.markdown("#### 🎨 Colorbar, spines & ticks")
                c1, c2 = st.columns(2)
                with c1:
                    show_cb = st.checkbox("Show colorbar (where applicable)",
                                           value=bool(fc.get('show_colorbar', True)),
                                           key="fc_cb")
                    cb_orient = st.selectbox("Colorbar orientation",
                                              ['vertical', 'horizontal'],
                                              index=0 if fc.get('colorbar_orientation',
                                                                 'vertical') == 'vertical' else 1,
                                              key="fc_cb_o")
                with c2:
                    sw = st.number_input("Spine width", value=float(fc.get('spine_width', 1.0)),
                                          min_value=0.1, max_value=5.0, step=0.1,
                                          key="fc_sw")
                    tw = st.number_input("Tick width", value=float(fc.get('tick_width', 1.0)),
                                          min_value=0.1, max_value=5.0, step=0.1,
                                          key="fc_tw")
                c1, c2, c3 = st.columns(3)
                with c1:
                    td = st.selectbox("Tick direction",
                                       ['out', 'in', 'inout'],
                                       index=['out', 'in', 'inout'].index(fc.get('tick_direction', 'out')),
                                       key="fc_td")
                with c2:
                    # v3.4.2: all int (StreamlitMixedNumericTypesError fix)
                    # v3.4.2: renamed key fc_tl → fc_tick_length to avoid
                    # collision with tight_layout checkbox (fc_tl was used twice).
                    _tl_val = fc.get('tick_length', 4)
                    try:
                        _tl_val = int(_tl_val) if _tl_val is not None else 4
                    except (TypeError, ValueError):
                        _tl_val = 4
                    tl = st.number_input("Tick length", value=_tl_val,
                                          min_value=0, max_value=20, step=1,
                                          key="fc_tick_length")

                if st.button("💾 Save figure customization", key="fc_save",
                              type="primary"):
                    def _parse(s):
                        s = s.strip() if isinstance(s, str) else s
                        if s is None or s == '':
                            return None
                        try:
                            return float(s)
                        except Exception:
                            return None
                    viz_cfg['figure_customization'] = {
                        'multi_field_width':    mf_w,
                        'multi_field_height':   mf_h,
                        'overlay_width':        ov_w,
                        'overlay_height':       ov_h,
                        'vector_width':         vc_w,
                        'vector_height':         vc_h,
                        'diagnostics_width':    dg_w,
                        'diagnostics_height':   dg_h,
                        'line_profile_width':   lp_w,
                        'line_profile_height':  lp_h,
                        'line_width':            lw_main,
                        'line_width_minor':      lw_minor,
                        'marker_size':           ms,
                        'marker_width':          mw,
                        'contour_line_width':    cw,
                        'quiver_width':          qw,
                        'quiver_scale':          qs,
                        'title_override':        ti_ov,
                        'xlabel_override':       xl_ov,
                        'ylabel_override':       yl_ov,
                        'show_grid':             show_grid,
                        'grid_alpha':            grid_alpha,
                        'xlim':                  (_parse(xmin_in), _parse(xmax_in)),
                        'ylim':                  (_parse(ymin_in), _parse(ymax_in)),
                        'show_colorbar':         show_cb,
                        'colorbar_orientation':  cb_orient,
                        'spine_width':           sw,
                        'tick_direction':        td,
                        'tick_width':            tw,
                        'tick_length':           tl,
                        'axis_equal':            axis_equal,
                        'tight_layout':          tight_layout,
                    }
                    st.success("✅ Figure customization saved. It will be "
                                "applied to all matplotlib figures in the "
                                "Results, Line Profiles, Plotly and Export tabs.")
                    st.rerun()

                st.markdown("---")
                st.caption("💡 **Tip**: Per-field colormap selection is in "
                            "the **Global & Field Scaling** tab — pick any "
                            f"of {len(ColormapRegistry.list_all())} colormaps "
                            "for each field independently.")

        # ---- Results -----------------------------------------------
        with tabs[3]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                solver = st.session_state.solver
                idx = st.slider("Frame", 0, len(history) - 1,
                                len(history) - 1, key="res_frame")
                vis = EnhancedTwinVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                style = {
                    'colormaps': {f: fset.get('cmap', FIELD_DEFAULT_CMAPS.get(f, 'viridis'))
                                   for f, fset in viz_cfg['field_settings'].items()},
                    'field_settings': viz_cfg['field_settings'],
                    'title_font_size': viz_cfg['title_font_size'],
                    'label_font_size': viz_cfg['label_font_size'],
                    'colorbar_fraction': viz_cfg['colorbar_fraction'],
                    'colorbar_pad': viz_cfg['colorbar_pad'],
                    'figure_customization': viz_cfg.get('figure_customization', {}),
                }

                mode_res = st.radio(
                    "Display mode",
                    ["Multi-field grid", "Overlay", "Vector field"],
                    horizontal=True, key="res_mode")

                # v3.4.9: enhanced inline figure customization per-mode
                # + LIVE per-field colormap picker for the Multi-field grid mode
                if mode_res == "Multi-field grid":
                    # Live per-field colormap picker (no Save button needed)
                    render_live_colormap_picker(
                        viz_cfg, key_prefix='res_cmap',
                        label='🎨 Per-field colormap (live, no Save)')
                    render_enhanced_figure_customization(
                        viz_cfg,
                        size_keys=('multi_field_width', 'multi_field_height'),
                        key_prefix='res_mf',
                        label='🎨 Multi-field grid — full customization',
                        show_line_styles=False,
                        show_series_colors=False,
                        show_legend=False,
                        show_spines=True,
                        show_font_sizes=True,
                        show_colorbar=True,
                        show_axis_limits=True,
                        show_text_overrides=True,
                        show_grid=True)
                elif mode_res == "Overlay":
                    render_enhanced_figure_customization(
                        viz_cfg,
                        size_keys=('overlay_width', 'overlay_height'),
                        key_prefix='res_ov',
                        label='🎨 Overlay — full customization',
                        show_line_styles=True,
                        show_series_colors=False,
                        show_legend=True,
                        show_spines=True,
                        show_font_sizes=True,
                        show_colorbar=True)
                else:
                    render_enhanced_figure_customization(
                        viz_cfg,
                        size_keys=('vector_width', 'vector_height'),
                        key_prefix='res_vc',
                        label='🎨 Vector field — full customization',
                        show_line_styles=True,
                        show_series_colors=False,
                        show_legend=True,
                        show_spines=True,
                        show_font_sizes=True,
                        show_colorbar=True)
                # rebuild style dict so viz_cfg['figure_customization'] changes take effect
                style['figure_customization'] = viz_cfg.get('figure_customization', {})
                style['colormaps'] = {f: fset.get('cmap', FIELD_DEFAULT_CMAPS.get(f, 'viridis'))
                                       for f, fset in viz_cfg['field_settings'].items()}
                style['field_settings'] = viz_cfg['field_settings']

                if mode_res == "Multi-field grid":
                    fig = vis.create_multi_field_comparison(history[idx], style)
                    if fig is not None:
                        st.pyplot(fig); plt.close(fig)
                elif mode_res == "Overlay":
                    ov = viz_cfg['overlay']
                    fig = vis.create_overlay(
                        history[idx],
                        base_field=ov['base'],
                        overlay_field=ov['overlay'],
                        alpha=ov['alpha'],
                        mode=ov['mode'],
                        n_contours=ov['n_contours'],
                        show_grain_ghost=ov['show_grain_ghost'],
                        grain_map=history[idx].get('grain_map'),
                        style_params=style)
                    if fig is not None:
                        st.pyplot(fig); plt.close(fig)
                    else:
                        st.warning("Overlay failed — check field selections.")
                else:
                    vv = viz_cfg['vector']
                    base_choice = ('sigma_h' if vv['mode'] == 'principal_stress'
                                   else 'sigma_eq')
                    fig = vis.create_vector_overlay(
                        history[idx],
                        source=vv['source'],
                        mode=vv['mode'],
                        density=vv['density'],
                        color=vv['color'],
                        scale=vv['scale'],
                        base_field=base_choice,
                        style_params=style)
                    if fig is not None:
                        st.pyplot(fig); plt.close(fig)
                    else:
                        st.warning("Vector field unavailable — need sxx/syy/sxy "
                                    "and a valid source.")

                st.subheader("Region-Averaged Diagnostics")
                # v3.4.10/12: quick size control — always visible, not inside expander.
                # This is the ONLY place the diagnostics width/height are edited.
                # The enhanced panel below has size_keys=None so it doesn't
                # duplicate these controls (clean separation of layout vs styling).
                fc_diag = viz_cfg.setdefault('figure_customization', default_figure_customization())
                st.markdown("**📐 Diagnostics figure size**")
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    # v3.4.11: use clamped wrapper to avoid ValueBelowMin/AboveMax
                    new_w = _clamped_number_input(
                        "Width (in)", value=fc_diag.get('diagnostics_width', 13.0),
                        min_value=4.0, max_value=30.0, step=0.5,
                        key="diag_quick_w",
                        help="Width of the 2×2 diagnostics figure in inches.")
                with c2:
                    new_h = _clamped_number_input(
                        "Height (in)", value=fc_diag.get('diagnostics_height', 8.0),
                        min_value=3.0, max_value=30.0, step=0.5,
                        key="diag_quick_h",
                        help="Height of the 2×2 diagnostics figure in inches.")
                with c3:
                    _cur_w = float(fc_diag.get('diagnostics_width', 13.0))
                    _cur_h = float(fc_diag.get('diagnostics_height', 8.0))
                    st.caption(
                        f"Current: {_cur_w:.1f} × {_cur_h:.1f} in → "
                        f"each subplot ≈ {_cur_w/2:.1f} × {_cur_h/2:.1f} in. "
                        f"For 2×2 grid, try **14×10** or **16×12** for "
                        f"less compression.")
                # v3.4.12: instant update — commit + rerun as soon as the
                # user changes the size (so they see the new figure immediately
                # without having to click elsewhere first).
                if (new_w != fc_diag.get('diagnostics_width') or
                        new_h != fc_diag.get('diagnostics_height')):
                    fc_diag['diagnostics_width'] = float(new_w)
                    fc_diag['diagnostics_height'] = float(new_h)
                    st.rerun()
                # v3.4.9/12: enhanced customization for diagnostics 2D line plots.
                # size_keys=None → hides size controls inside the expander
                # (they're now in the quick size block above, avoiding duplication).
                render_enhanced_figure_customization(
                    viz_cfg,
                    size_keys=None,
                    key_prefix='diag',
                    label='🎨 Diagnostics — full 2D line customization',
                    show_line_styles=True,
                    show_series_colors=True,  # per-series colors for c_top/c_bot/Avg/Max/Mass
                    show_legend=True,
                    show_spines=True,
                    show_font_sizes=True,
                    show_colorbar=False,
                    show_axis_limits=False,
                    show_text_overrides=True,
                    show_grid=True,
                    n_series=5)
                t_disp, t_unit = fmt_time_range(solver.history['time'], params)
                t = t_disp
                fig_cfg = viz_cfg.get('figure_customization', {})
                # v3.4.9: per-series line style/marker/color
                lw_main = float(fig_cfg.get('line_width', 2.0))
                lw_minor = float(fig_cfg.get('line_width_minor', 1.5))
                ls_main = fig_cfg.get('line_style', '-')
                ls_minor = fig_cfg.get('line_style_minor', '--')
                ms_main = fig_cfg.get('marker_style', 'none')
                ms_minor = fig_cfg.get('marker_style_minor', 'none')
                marker_size = float(fig_cfg.get('marker_size', 6))
                marker_ew = float(fig_cfg.get('marker_width', 1.0))
                col1 = fig_cfg.get('color_series_1', '#1f77b4')  # Poly-Cu
                col2 = fig_cfg.get('color_series_2', '#d62728')  # nc-Cu
                col3 = fig_cfg.get('color_series_3', '#2ca02c')  # Avg
                col4 = fig_cfg.get('color_series_4', '#2ca02c')  # Max
                col5 = fig_cfg.get('color_series_5', '#000000')  # Mass
                # matplotlib marker = '' means "no marker"
                m_main = '' if ms_main in ('none', '') else ms_main
                m_minor = '' if ms_minor in ('none', '') else ms_minor
                # matplotlib linestyle = '' or 'none' means "no line"
                l_main = '' if ls_main in ('none', '') else ls_main
                l_minor = '' if ls_minor in ('none', '') else ls_minor
                grid_alpha = float(fig_cfg.get('grid_alpha', 0.3))
                show_grid = bool(fig_cfg.get('show_grid', True))
                fig_w = float(fig_cfg.get('diagnostics_width', 13.0))
                fig_h = float(fig_cfg.get('diagnostics_height', 8.0))
                t_xlabel = fig_cfg.get('xlabel_override', '').strip() or f't ({t_unit})'

                # v3.4.13: Use GridSpec with explicit height_ratios=[1, 1]
                # to LOCK row heights at 50/50 regardless of the twinx() on
                # the bottom-right subplot (Mass conservation).
                # Previous approach (constrained_layout=True) was buggy:
                # constrained_layout saw the extra y-axis labels on the
                # twin axis and dynamically resized that cell — at the
                # expense of the top row, squashing it. GridSpec with
                # explicit height_ratios prevents this dynamic resizing.
                fig = plt.figure(figsize=(fig_w, fig_h))
                gs = GridSpec(2, 2, figure=fig,
                              height_ratios=[1, 1],
                              hspace=0.35, wspace=0.30)
                axes = np.array([
                    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
                ])
                axes[0, 0].plot(t, solver.history['c_top'], color=col1,
                                 linestyle=l_main, lw=lw_main, marker=m_main,
                                 ms=marker_size, mew=marker_ew,
                                 label='Poly-Cu (top)')
                axes[0, 0].plot(t, solver.history['c_bot'], color=col2,
                                 linestyle=l_main, lw=lw_main, marker=m_main,
                                 ms=marker_size, mew=marker_ew,
                                 label='nc-Cu (bottom)')
                axes[0, 0].set_xlabel(t_xlabel)
                axes[0, 0].set_ylabel('mean c (Au)')
                axes[0, 0].set_title('Au concentration: Poly-Cu vs nc-Cu')
                apply_axis_styling(axes[0, 0], fig_cfg,
                                    title='Au concentration: Poly-Cu vs nc-Cu',
                                    xlabel=t_xlabel, ylabel='mean c (Au)',
                                    force_aspect='auto')
                apply_legend_styling(axes[0, 0], fig_cfg)

                axes[0, 1].plot(t, solver.history['gb_top'], color=col1,
                                 linestyle=l_main, lw=lw_main, marker=m_main,
                                 ms=marker_size, mew=marker_ew,
                                 label='Poly-Cu')
                axes[0, 1].plot(t, solver.history['gb_bot'], color=col2,
                                 linestyle=l_main, lw=lw_main, marker=m_main,
                                 ms=marker_size, mew=marker_ew,
                                 label='nc-Cu')
                apply_axis_styling(axes[0, 1], fig_cfg,
                                    title='GB density (Hart indicator)',
                                    xlabel=t_xlabel, ylabel='mean φ_gb',
                                    force_aspect='auto')
                apply_legend_styling(axes[0, 1], fig_cfg)

                axes[1, 0].plot(t, np.array(solver.history['avg_stress'])/1e9,
                                 color=col3, linestyle=l_main, lw=lw_main,
                                 marker=m_main, ms=marker_size, mew=marker_ew,
                                 label='Avg')
                axes[1, 0].plot(t, np.array(solver.history['max_stress'])/1e9,
                                 color=col4, linestyle=l_minor, lw=lw_minor,
                                 marker=m_minor, ms=marker_size, mew=marker_ew,
                                 label='Max')
                apply_axis_styling(axes[1, 0], fig_cfg,
                                    title='Stress evolution',
                                    xlabel=t_xlabel, ylabel='σ (GPa)',
                                    force_aspect='auto')
                apply_legend_styling(axes[1, 0], fig_cfg)

                au_mass = np.array(solver.history['au_mass'])
                axes[1, 1].plot(t, au_mass / max(au_mass[0], 1e-12),
                                 color=col5, linestyle=l_main, lw=lw_main,
                                 marker=m_main, ms=marker_size, mew=marker_ew,
                                 label='Au mass / initial')
                c_top_arr = np.array(solver.history['c_top'])
                c_bot_arr = np.array(solver.history['c_bot'])
                ax2 = axes[1, 1].twinx()
                ax2.plot(t, c_top_arr, color=col1, linestyle=l_minor,
                          lw=lw_minor, marker=m_minor, ms=marker_size,
                          mew=marker_ew, label='c_top')
                ax2.plot(t, c_bot_arr, color=col2, linestyle=l_minor,
                          lw=lw_minor, marker=m_minor, ms=marker_size,
                          mew=marker_ew, label='c_bot')
                ax2.set_ylabel('mean c')
                apply_axis_styling(axes[1, 1], fig_cfg,
                                    title='Mass conservation (guarded)',
                                    xlabel=t_xlabel, ylabel='mass ratio',
                                    force_aspect='auto')
                apply_legend_styling(axes[1, 1], fig_cfg)

                # v3.4.13: do NOT call tight_layout() or constrained_layout.
                # GridSpec (above) explicitly sets height_ratios=[1,1] and
                # hspace/wspace — the layout is fully determined. Calling
                # tight_layout on top of GridSpec would re-introduce the
                # twin-axis squashing bug (it gives the twinx subplot
                # disproportionate space).
                st.pyplot(fig); plt.close(fig)

        # ---- Line profiles -----------------------------------------
        with tabs[4]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                profiler = EnhancedLineProfiler(Nx, Ny, dx, dy)

                # v3.4.9: enhanced customization for line-profile figures
                render_enhanced_figure_customization(
                    viz_cfg,
                    size_keys=('line_profile_width', 'line_profile_height'),
                    key_prefix='lp',
                    label='🎨 Line profile — full customization',
                    show_line_styles=True,
                    show_series_colors=False,
                    show_legend=True,
                    show_spines=True,
                    show_font_sizes=True,
                    show_colorbar=False,
                    show_axis_limits=True,
                    show_text_overrides=True,
                    show_grid=True)

                sub_a, sub_b = st.tabs(["Single frame", "Multi-time compare"])

                with sub_a:
                    idx = st.slider("Frame", 0, len(history) - 1,
                                    len(history) - 1, key="lp_frame")
                    res = history[idx]
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="lp_field")
                    ptype = st.selectbox("Profile direction",
                                          ["horizontal", "vertical",
                                           "diagonal", "anti_diagonal"],
                                          key="lp_ptype")
                    pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                    key="lp_pos")

                    d, unit = _rescale_field(field, res[field])
                    out = profiler.extract_profile(d, ptype, pos)
                    if out is FAILED or out is None:
                        st.warning("Profile extraction failed.")
                    else:
                        dist, prof, _ = out
                        if dist is not None:
                            fc = viz_cfg.get('figure_customization', {})
                            fig_w = float(fc.get('line_profile_width', 11.0))
                            fig_h = float(fc.get('line_profile_height', 4.0))
                            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                            # v3.4.9: honor line_style + marker_style + series color
                            ls_main = fc.get('line_style', '-')
                            ms_main = fc.get('marker_style', 'none')
                            m_main = '' if ms_main in ('none', '') else ms_main
                            l_main = '' if ls_main in ('none', '') else ls_main
                            s1_color = fc.get('color_series_1', '#1f77b4')
                            ax.plot(dist, prof, color=s1_color,
                                     linestyle=l_main,
                                     lw=float(fc.get('line_width', 2.0)),
                                     marker=m_main,
                                     ms=float(fc.get('marker_size', 6)),
                                     mew=float(fc.get('marker_width', 1.0)))
                            apply_axis_styling(ax, fc,
                                                title=f'{field} · {ptype} profile · frame {idx}',
                                                xlabel='Position (nm)',
                                                ylabel=f'{field} {unit}',
                                                force_aspect='auto')
                            st.pyplot(fig); plt.close(fig)

                with sub_b:
                    field_mt = st.selectbox("Field",
                                             list(FIELD_DEFAULT_CMAPS.keys()),
                                             index=0, key="lp_mt_field")
                    ptype_mt = st.selectbox("Profile direction",
                                             ["horizontal", "vertical",
                                              "diagonal", "anti_diagonal"],
                                             key="lp_mt_ptype")
                    pos_mt = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                        key="lp_mt_pos")
                    default_frames = list(range(0, len(history),
                                                 max(1, len(history) // 6)))
                    frames_sel = st.multiselect(
                        "Frames",
                        list(range(len(history))),
                        default=default_frames)
                    use_plotly_mt = st.checkbox("Use Plotly (interactive)",
                                                 value=True, key="lp_mt_plotly")
                    if frames_sel:
                        vis_lp = EnhancedTwinVisualizer(Nx, Ny, dx, dy,
                                                          dt=params['dt'])
                        fig_mt = vis_lp.create_multi_time_profiles(
                            history, field_mt, ptype_mt, frames_sel,
                            position_ratio=pos_mt, use_plotly=use_plotly_mt,
                            viz_cfg=viz_cfg)
                        if fig_mt is None:
                            st.warning("No profiles could be extracted.")
                        elif use_plotly_mt:
                            st.plotly_chart(fig_mt, use_container_width=True)
                        else:
                            st.pyplot(fig_mt); plt.close(fig_mt)

        # ---- Plotly -------------------------------------------------
        with tabs[5]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                vis = EnhancedTwinVisualizer(Nx, Ny, dx, dy, dt=params['dt'])

                view = st.radio("Plotly view",
                                 ["Heatmap (frame)", "3D surface",
                                  "Time-scrubbing animation",
                                  "Line profiles"],
                                 horizontal=True, key="px_view")

                # v3.4.6: inline customization for Plotly figures
                # (overlay_width/height controls Plotly heatmap; line_profile
                # size controls Plotly line profiles)
                if view == "Heatmap (frame)" or view == "Time-scrubbing animation":
                    _sz_keys = ('overlay_width', 'overlay_height')
                    _lbl = 'Plotly heatmap customization (width × height)'
                elif view == "3D surface":
                    _sz_keys = None  # 3D surface has fixed 800×600 in code
                    _lbl = 'Plotly 3D surface customization (text + ticks only)'
                else:
                    _sz_keys = ('line_profile_width', 'line_profile_height')
                    _lbl = 'Plotly line-profile customization'
                render_inline_figure_customization(
                    viz_cfg,
                    size_keys=_sz_keys,
                    key_prefix=f'px_{view[:3]}',
                    label=_lbl,
                    show_lines=(view == "Line profiles"),
                    show_colorbar=(view != "Line profiles"))

                if view == "Heatmap (frame)":
                    idx = st.slider("Frame", 0, len(history) - 1,
                                    len(history) - 1, key="px_frame")
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="px_field")
                    cmap = viz_cfg['field_settings'].get(field, {}).get(
                        'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                    fig_h = vis.create_plotly_heatmap(
                        history[idx], field, idx, cmap, viz_cfg=viz_cfg)
                    if fig_h is not None:
                        st.plotly_chart(fig_h, use_container_width=True)

                elif view == "3D surface":
                    idx = st.slider("Frame", 0, len(history) - 1,
                                    len(history) - 1, key="px_3d_frame")
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="px_3d_field")
                    show_c = st.checkbox("Show contours", value=True,
                                          key="px_3d_cont")
                    cmap = viz_cfg['field_settings'].get(field, {}).get(
                        'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                    fig_3d = vis.create_plotly_3d_surface(
                        history[idx], field, idx, cmap,
                        show_contours=show_c, viz_cfg=viz_cfg)
                    if fig_3d is not None:
                        st.plotly_chart(fig_3d, use_container_width=True)

                elif view == "Time-scrubbing animation":
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="px_anim_field")
                    cmap = viz_cfg['field_settings'].get(field, {}).get(
                        'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                    with st.spinner("Building animation frames..."):
                        fig_anim = vis.create_plotly_animation(
                            history, field, params['dt'], cmap, viz_cfg=viz_cfg)
                    if fig_anim is not None:
                        st.plotly_chart(fig_anim, use_container_width=True)
                    else:
                        st.warning("Could not build animation.")

                else:
                    idx = st.slider("Frame", 0, len(history) - 1,
                                    len(history) - 1, key="px_lp_frame")
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="px_lp_field")
                    ptypes = st.multiselect(
                        "Directions",
                        ["horizontal", "vertical", "diagonal", "anti_diagonal"],
                        default=["horizontal", "vertical"], key="px_lp_types")
                    pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                    key="px_lp_pos")
                    if ptypes:
                        fig_lp = vis.create_plotly_line_profiles(
                            history[idx], field, ptypes, pos, viz_cfg=viz_cfg)
                        if fig_lp is not None:
                            st.plotly_chart(fig_lp, use_container_width=True)

        # ---- Export -------------------------------------------------
        with tabs[6]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                params = st.session_state.params
                history = st.session_state.history
                sim_data = {'metadata':
                            {'created_at': datetime.now().isoformat()}}
                name = (f"au_cu_bonding_Nx{params['Nx']}_Ny{params['Ny']}"
                        f"_npoly{params['n_poly_grains']}_seed{params['seed']}")

                st.subheader("📦 Data export")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Pickle"):
                        buf, fname = DataExporter.export_pkl(
                            sim_data, params, history, name)
                        if buf:
                            st.download_button("Download .pkl", buf, fname)
                    if st.button("JSON"):
                        buf, fname = DataExporter.export_json(
                            sim_data, params, history, name)
                        if buf:
                            st.download_button("Download .json", buf, fname)
                    if st.button("PyTorch (.pt)"):
                        buf, fname = DataExporter.export_pt(
                            sim_data, params, history, name)
                        if buf:
                            st.download_button("Download .pt", buf, fname)
                with col2:
                    if st.button("CSV (zip)"):
                        extent = [-Nx*dx/2, Nx*dx/2, -Ny*dy/2, Ny*dy/2]
                        buf, fname = DataExporter.export_csv(
                            history, name, extent, Nx, Ny, dx, dy)
                        if buf:
                            st.download_button("Download .zip", buf, fname)
                    if st.button("HDF5"):
                        buf, fname = DataExporter.export_hdf5(
                            sim_data, params, history, name, Nx, Ny, dx, dy)
                        if buf:
                            st.download_button("Download .h5", buf, fname)

                st.markdown("---")
                st.subheader("🎬 Animation export")
                an = viz_cfg['animation']
                anim_field = st.selectbox("Field",
                                           list(FIELD_DEFAULT_CMAPS.keys()),
                                           index=0, key="anim_field")
                c1, c2, c3 = st.columns(3)
                with c1:
                    a_fmt = st.selectbox("Format", ['gif', 'mp4'],
                                          index=0 if an['format'] == 'gif' else 1,
                                          key="anim_fmt_ui")
                with c2:
                    a_fps = st.slider("FPS", 1, 30, int(an['fps']), 1,
                                       key="anim_fps_ui")
                with c3:
                    a_dpi = st.slider("DPI", 72, 300, int(an['dpi']), 12,
                                       key="anim_dpi_ui")
                a_wm = st.checkbox("Watermark",
                                    value=an['watermark'], key="anim_wm_ui")
                a_wm_text = st.text_input("Watermark text",
                                           value=an['watermark_text'],
                                           key="anim_wm_text_ui")
                wm_pos_list_ui = ['lower right', 'lower left', 'upper right', 'upper left']
                wm_idx_ui = (wm_pos_list_ui.index(an['watermark_pos'])
                             if an['watermark_pos'] in wm_pos_list_ui else 0)
                a_wm_pos = st.selectbox("Watermark position",
                                         wm_pos_list_ui,
                                         index=wm_idx_ui,
                                         key="anim_wm_pos_ui")

                if st.button("🎬 Generate single-field animation"):
                    fs_anim = viz_cfg['field_settings'].get(anim_field, {})
                    cmap = fs_anim.get('cmap', FIELD_DEFAULT_CMAPS.get(anim_field, 'viridis'))
                    vis_exp = EnhancedTwinVisualizer(Nx, Ny, dx, dy,
                                                      dt=params['dt'])
                    buf = vis_exp.create_animation_buffer(
                        history, anim_field, params['dt'],
                        output_format=a_fmt, fps=a_fps, dpi=a_dpi,
                        cmap_name=cmap, watermark=a_wm,
                        watermark_text=a_wm_text,
                        watermark_pos=a_wm_pos, viz_cfg=viz_cfg)
                    if buf is not None:
                        st.download_button(
                            f"Download .{a_fmt}", buf,
                            f"{name}_{anim_field}.{a_fmt}")
                    else:
                        st.error("Animation generation failed. For MP4, "
                                  "ensure ffmpeg is installed.")

                if st.button("📦 Batch generate ALL fields (zip of GIFs)"):
                    vis_exp = EnhancedTwinVisualizer(Nx, Ny, dx, dy,
                                                      dt=params['dt'])
                    zip_buf = BytesIO()
                    with zipfile.ZipFile(zip_buf, "w",
                                          zipfile.ZIP_DEFLATED) as zf:
                        prog = st.progress(0)
                        fields_all = list(FIELD_DEFAULT_CMAPS.keys())
                        for k, fname in enumerate(fields_all):
                            fs_b = viz_cfg['field_settings'].get(fname, {})
                            cmap = fs_b.get('cmap', FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
                            buf = vis_exp.create_animation_buffer(
                                history, fname, params['dt'],
                                output_format='gif', fps=a_fps, dpi=a_dpi,
                                cmap_name=cmap, watermark=a_wm,
                                watermark_text=a_wm_text,
                                watermark_pos=a_wm_pos, viz_cfg=viz_cfg)
                            if buf is not None:
                                zf.writestr(f"{name}_{fname}.gif",
                                            buf.getvalue())
                            prog.progress((k + 1) / len(fields_all))
                    zip_buf.seek(0)
                    st.download_button("Download all GIFs (.zip)", zip_buf,
                                        f"{name}_all_fields.zip")

        # ============================================================
        # v3.4.15-sustainability: 🌱 SUSTAINABILITY ANALYSIS TAB
        # ============================================================
        # Quantifies the energy / throughput / Au-utilization advantage of
        # the nc-Cu bottom layer vs. a symmetric Poly-Cu / Au / Poly-Cu
        # reference configuration.
        #
        # The argument collapses onto one geometric quantity: the
        # grain-boundary volume fraction φ_gb ≈ δ / d. Because the Hart
        # effective diffusivity D_eff ≈ D_gb · φ_gb, replacing a 15 nm
        # nc-Cu layer with 150 nm Poly-Cu reduces D_eff by ~10× — and so
        # the time / temperature / pressure required to reach the same
        # bonding quality must go UP by the corresponding factor.
        with tabs[7]:
            st.header("🌱 Sustainability Analysis (Data-Driven)")
            st.caption(
                "Select saved simulations to compute MEASURED sustainability "
                "metrics. Substrate type is auto-classified by average grain "
                "size: d < 100 nm = Nanocrystalline (nc-Cu); d >= 100 nm = "
                "Polycrystalline (Poly-Cu). v3.4.17 decouples this tab from "
                "the active in-memory solver — all kinetics now come "
                "from saved 1D diagnostics (`solver.history`) stored in the "
                "SimulationDatabase, so you can compare runs across sessions "
                "and grain sizes without re-running. Workflow: (1) set bottom "
                "layer = nc-Cu, run, save; (2) set bottom layer = Poly-Cu "
                "(reference), run, save; (3) select both below.")

            sims_db = SimulationDatabase.all()
            if not sims_db:
                st.info(
                    "Run and save simulations first to enable data-driven "
                    "sustainability analysis. The Sustainability tab now "
                    "reads saved 1D diagnostics from the SimulationDatabase "
                    "instead of the active in-memory solver.")
            else:
                # ---- Build selection list with auto-classification ----
                sim_options = []
                for sid, sim in sims_db.items():
                    d = float(sim['params'].get('nc_grain_size_nm', 0))
                    sub_type = classify_substrate(d)
                    diag = sim.get('diagnostics', {})
                    c_bot_arr = np.array(diag.get('c_bot', [0]))
                    c_max = float(np.max(c_bot_arr)) if c_bot_arr.size else 0.0
                    has_diag = bool(diag and diag.get('time'))
                    tag = "OK" if has_diag else "no-diag"
                    sim_options.append({
                        'id': sid,
                        'label': (f"{sid[:6]}… | d={d:.0f} nm ({sub_type}) | "
                                  f"max<c>={c_max:.3f} | {tag}"),
                        'd': d,
                        'sim': sim,
                        'has_diag': has_diag,
                    })

                # ---- 1. Select configurations ----
                st.subheader("1. Select configurations")
                c1, c2 = st.columns(2)
                with c1:
                    ref_idx = st.selectbox(
                        "Reference (baseline)",
                        range(len(sim_options)),
                        format_func=lambda i: sim_options[i]['label'],
                        key="sus_ref_idx",
                        help="Select the Poly-Cu or larger-grain reference "
                             "(d >= 100 nm). This is the slower configuration.")
                with c2:
                    design_idx = st.selectbox(
                        "Design (nc-Cu)",
                        range(len(sim_options)),
                        format_func=lambda i: sim_options[i]['label'],
                        key="sus_design_idx",
                        help="Select the nanocrystalline design (d < 100 nm). "
                             "This is the faster, as-designed configuration.")

                ref_data = sim_options[ref_idx]
                design_data = sim_options[design_idx]

                if ref_idx == design_idx:
                    st.warning(
                        "⚠️ Reference and Design are the same "
                        "simulation. S_t will be 1.0x (no penalty). Select "
                        "two different runs for a meaningful comparison.")

                if not ref_data['has_diag'] or not design_data['has_diag']:
                    st.warning(
                        "⚠️ One or both selected simulations have "
                        "no saved 1D diagnostics (time-series). This happens "
                        "for runs saved before v3.4.17. Re-run and save to "
                        "populate the diagnostics. Metrics below will fall "
                        "back to geometric S_t (Hart linear-scaling).")

                # Use the design sim's params as the calculation context
                params = dict(design_data['sim']['params'])
                t_unit = get_time_unit_label(params)
                d_current = float(design_data['d'])
                d_ref_sel = float(ref_data['d'])
                gb_width = float(params.get('gb_width_nm', 1.5))

                # ---- 2. Material constants (Arrhenius + uncertainty) ----
                st.subheader("2. Material constants (Arrhenius + uncertainty)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    Q_gb_kj = st.number_input(
                        "Q_gb (kJ/mol)", value=100.0, min_value=20.0,
                        max_value=400.0, step=5.0, key="sus_q_gb",
                        help="Grain-boundary diffusion activation energy for Au in Cu. "
                             "Literature range 80-120 kJ/mol.")
                    Q_gb_J = float(Q_gb_kj) * 1e3
                with c2:
                    Q_lat_kj = st.number_input(
                        "Q_lattice (kJ/mol)", value=200.0, min_value=40.0,
                        max_value=400.0, step=5.0, key="sus_q_lat",
                        help="Lattice diffusion activation energy for Au in Cu. "
                             "Literature range 180-220 kJ/mol. Used together "
                             "with Q_gb in the effective-Q calculation.")
                    Q_lat_J = float(Q_lat_kj) * 1e3
                with c3:
                    sigma_Q_kj = st.number_input(
                        "σ_Q (kJ/mol, lit. uncertainty)", value=20.0,
                        min_value=0.0, max_value=60.0, step=1.0, key="sus_sigma_q",
                        help="1-sigma literature uncertainty on Q_gb. Used for "
                             "the Monte-Carlo 95% CI on dT. +/-20 kJ/mol is standard.")
                    sigma_Q_J = float(sigma_Q_kj) * 1e3
                T_bond_K = st.number_input(
                    "T_bond (K)", value=523.0, min_value=300.0, max_value=800.0,
                    step=5.0, format="%.1f", key="sus_t_bond",
                    help="Bonding temperature in Kelvin. 250 C = 523 K (baseline).")

                # Optional override of d_ref (defaults to the reference sim's d)
                d_ref = st.number_input(
                    "Reference grain size d_ref (nm)",
                    value=d_ref_sel, min_value=50.0, max_value=2000.0,
                    step=10.0, format="%.1f", key="sus_d_ref",
                    help="Grain size of the Poly-Cu reference. Defaults to the "
                         "selected reference simulation's grain size. Override "
                         "only if you want to explore a hypothetical d_ref.")

                # ---- 3. Bonding-quality target ----
                st.subheader("3. Bonding-quality target")
                target_c = st.slider(
                    "Target <c>_Ob* (Au utilization threshold)",
                    0.01, 0.20, 0.06, 0.005,
                    help="Region-averaged Au concentration in the bottom "
                         "layer required to consider the joint well-bonded. "
                         "0.06 = 77%-of-max threshold from prior phase-field runs.")

                # ---- Cost model ----
                with st.expander("Cost-model inputs (energy + machine + material)",
                                  expanded=False):
                    st.caption(
                        "These feed `compute_cost_per_joint()` and the Pareto plot. "
                        "All costs are computed in $/joint at the batch size below.")
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        furnace_kW = st.number_input(
                            "Furnace power (kW)", value=2.0, min_value=0.1,
                            max_value=50.0, step=0.1, format="%.1f",
                            key="sus_furnace")
                        grid_g = st.number_input(
                            "Grid carbon (kg CO2/kWh)", value=0.40, min_value=0.0,
                            max_value=1.5, step=0.05, format="%.2f",
                            key="sus_grid_g",
                            help="European grid ~0.40; US ~0.42; coal ~0.85; renewable ~0.05.")
                    with cc2:
                        batch_joints = st.number_input(
                            "Batch size (joints)", value=100, min_value=1,
                            max_value=10000, step=10, key="sus_batch")
                        energy_price = st.number_input(
                            "Energy price ($/kWh)", value=0.10, min_value=0.0,
                            max_value=1.0, step=0.01, format="%.2f",
                            key="sus_energy_price")
                        machine_hourly = st.number_input(
                            "Machine hourly ($/h)", value=50.0, min_value=0.0,
                            max_value=500.0, step=5.0, key="sus_machine_hourly")
                    with cc3:
                        cu_mass_kg = st.number_input(
                            "Cu mass per batch (kg)", value=1e-3, min_value=1e-6,
                            max_value=1.0, step=1e-4, format="%.4f",
                            key="sus_cu_mass")
                        au_mass_kg = st.number_input(
                            "Au mass per batch (kg, optional)", value=0.0,
                            min_value=0.0, max_value=0.1, step=1e-5, format="%.5f",
                            key="sus_au_mass")
                        nano_premium = st.number_input(
                            "nc-Cu nano-premium ($/kg)", value=50.0, min_value=0.0,
                            max_value=1000.0, step=5.0, key="sus_nano_prem",
                            help="Premium over bulk Cu for nc-Cu powder at d=d_current. "
                                 "Decays exponentially with d in the Pareto plot.")

                # =====================================================
                # EXTRACT DIAGNOSTICS FROM SAVED SIMS
                # =====================================================
                ref_diag = ref_data['sim'].get('diagnostics', {})
                design_diag = design_data['sim'].get('diagnostics', {})

                t_ref = np.array(ref_diag.get('time', []))
                c_ref = np.array(ref_diag.get('c_bot', []))
                t_des = np.array(design_diag.get('time', []))
                c_des = np.array(design_diag.get('c_bot', []))

                # =====================================================
                # COMPUTE MEASURED TIME-TO-TARGET
                # =====================================================
                t_des_reach, des_reached = _time_to_target(t_des, c_des, target_c)
                t_ref_reach, ref_reached = _time_to_target(t_ref, c_ref, target_c)

                # =====================================================
                # 4. MEASURED BONDING KINETICS
                # =====================================================
                st.markdown("---")
                st.subheader("4. Measured bonding kinetics")
                st.caption(
                    f"Design: d={design_data['d']:.0f} nm "
                    f"({classify_substrate(design_data['d'])})  |  "
                    f"Reference: d={ref_data['d']:.0f} nm "
                    f"({classify_substrate(ref_data['d'])}).  "
                    f"Curves are pulled directly from the saved 1D diagnostics "
                    f"of the two selected simulations. No geometric proxy.")

                if t_des.size > 0 or t_ref.size > 0:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    if t_des.size > 0:
                        ax.plot(t_des, c_des, 'r-', lw=2.5,
                                label=f"Design: d={design_data['d']:.0f} nm "
                                      f"({classify_substrate(design_data['d'])})")
                    if t_ref.size > 0:
                        ax.plot(t_ref, c_ref, 'b--', lw=2.5,
                                label=f"Reference: d={ref_data['d']:.0f} nm "
                                      f"({classify_substrate(ref_data['d'])})")
                    ax.axhline(target_c, color='k', ls=':', lw=1.5,
                               label=f"Target {target_c:.3f}")
                    if des_reached:
                        ax.plot(t_des_reach, target_c, 'ro', ms=12, zorder=5)
                        ax.annotate(f"t_reach={t_des_reach:.2f}",
                                    (t_des_reach, target_c),
                                    textcoords="offset points", xytext=(0, 12),
                                    ha='center', color='red')
                    if ref_reached:
                        ax.plot(t_ref_reach, target_c, 'bs', ms=12, zorder=5)
                        ax.annotate(f"t_reach={t_ref_reach:.2f}",
                                    (t_ref_reach, target_c),
                                    textcoords="offset points", xytext=(0, 12),
                                    ha='center', color='blue')
                    ax.set_xlabel(f"Time ({t_unit})")
                    ax.set_ylabel("Region-averaged <c>_Ob")
                    ax.set_title("Bonding kinetics — measured from saved diagnostics")
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.info("No trajectory data available in the selected "
                            "simulations' diagnostics.")

                # =====================================================
                # 5. SUSTAINABILITY METRICS (MEASURED)
                # =====================================================
                st.markdown("---")
                st.subheader("5. Sustainability metrics (measured)")

                if des_reached and ref_reached and t_des_reach > 1e-9:
                    S_t_measured = t_ref_reach / t_des_reach

                    # Build a paired_results mock from the saved diagnostics
                    # so compute_sustainability_metrics uses the MEASURED
                    # t_reach values instead of the geometric time-stretch.
                    paired_results_mock = {
                        'nc': {
                            't_reached': float(t_des_reach),
                            'reached': True,
                            'c_bot': c_des,
                            'time': t_des,
                        },
                        'poly': {
                            't_reached': float(t_ref_reach),
                            'reached': True,
                            'c_bot': c_ref,
                            'time': t_ref,
                            'S_t_measured': float(S_t_measured),
                        },
                    }

                    # Ensure the design params reflect the selected design sim
                    params_for_calc = dict(design_data['sim']['params'])

                    metrics = compute_sustainability_metrics(
                        solver_history=design_diag,
                        params=params_for_calc,
                        target_c=target_c,
                        d_current=d_current,
                        d_ref=d_ref,
                        gb_width=gb_width,
                        Q_gb_J=Q_gb_J,
                        Q_lat_J=Q_lat_J,
                        T_bond_K=T_bond_K,
                        applied_stress_MPa=(
                            float(params_for_calc.get('applied_stress', 30e6)) / 1e6),
                        furnace_kW=furnace_kW,
                        grid_carbon_kg_per_kWh=grid_g,
                        batch_joints=int(batch_joints),
                        energy_price_usd_kwh=energy_price,
                        machine_hourly_usd=machine_hourly,
                        cu_mass_kg=cu_mass_kg,
                        au_mass_kg=(au_mass_kg if au_mass_kg > 0 else None),
                        nano_premium_usd_kg=nano_premium,
                        sigma_Q_J=sigma_Q_J,
                        paired_results=paired_results_mock)

                    # ---- Primary metric cards ----
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "Thermal-budget multiplier S_t (Measured)",
                            f"{S_t_measured:.2f}x",
                            delta=f"Reference is {S_t_measured:.1f}x slower",
                            delta_color="inverse",
                            help="Measured S_t = t_ref_reach / t_des_reach, "
                                 "computed from the saved diagnostics of the "
                                 "two selected simulations. No geometric proxy.")
                    with c2:
                        dT_lo, dT_hi = metrics['Delta_T_95CI']
                        dT_med = metrics['Delta_T_median']
                        if np.isfinite(dT_med) and dT_med > 0:
                            dT_label = f"+{metrics['Delta_T_eff']:.0f} C"
                            dT_delta = (f"95% CI: [{dT_lo:+.0f}, {dT_hi:+.0f}] C "
                                        f"(median {dT_med:+.0f})")
                        else:
                            dT_label = f"+{metrics['Delta_T_eff']:.0f} C"
                            dT_delta = f"Q_eff = {metrics['Q_eff_poly_kJmol']:.1f} kJ/mol"
                        st.metric("Temperature penalty ΔT (Q_eff, exact Arrhenius)",
                                   dT_label, delta=dT_delta, delta_color="inverse")
                    with c3:
                        st.metric("Pressure penalty S_P",
                                   f"{metrics['S_P']:.1f}x",
                                   delta=(f"poly: ~{metrics['sigma_poly_MPa']:.0f} MPa vs "
                                          f"nc: {metrics['sigma_nc_MPa']:.0f} MPa"),
                                   delta_color="inverse")
                    with c4:
                        cs = metrics['composite_score']
                        st.metric("Composite MCDA Σ",
                                   f"{cs:.2f} / 1.00",
                                   delta=f"{cs*100:.0f}% of Poly-Cu headroom",
                                   delta_color="off",
                                   help="Geometric-mean MCDA over 5 axes (energy, CO2, "
                                        "throughput, Au utilisation, cost). 1.0 = as good "
                                        "as Poly-Cu; <1.0 = nc-Cu advantage.")

                    st.caption(
                        "Metrics computed using MEASURED t_reach from the saved "
                        "diagnostics of the two selected simulations. Geometric "
                        "Hart-formula proxies are overridden by actual phase-field "
                        "kinetics.")

                    # ---- Geometric lever + effective Q ----
                    st.markdown("---")
                    st.subheader("Geometric lever: φ_gb and effective Q")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Design grain size d", f"{d_current:.1f} nm")
                        st.metric("Design φ_gb ≈ δ/d", f"{metrics['phi_gb_nc']:.3f}")
                    with c2:
                        st.metric("Reference grain size d_ref", f"{d_ref:.1f} nm")
                        st.metric("Reference φ_gb ≈ δ/d_ref", f"{metrics['phi_gb_poly']:.3f}")
                    with c3:
                        st.metric("GB width δ", f"{gb_width:.2f} nm")
                        st.metric("D_eff ratio (Design/Ref)",
                                  f"{metrics['D_eff_nc']/max(metrics['D_eff_poly'],1e-12):.1f}x")
                    with c4:
                        st.metric("Q_eff (Design)", f"{metrics['Q_eff_nc_kJmol']:.1f} kJ/mol")
                        st.metric("Q_eff (Reference)", f"{metrics['Q_eff_poly_kJmol']:.1f} kJ/mol")
                    st.caption(
                        f"ΔT uses Q_eff (D-weighted average of Q_lat and "
                        f"Q_gb) instead of Q_gb alone. With D_lat="
                        f"{params_for_calc.get('D_lattice',0.01):.3f} and D_gb="
                        f"{params_for_calc.get('D_gb',50.0):.1f}, the GB term "
                        f"D_gb*φ_gb dominates for both configurations, so "
                        f"Q_eff ≈ Q_gb in this regime — but the formula "
                        f"only reveals this if it is implemented correctly.")

                    # ---- Engineering metrics table ----
                    st.markdown("---")
                    st.subheader("Engineering metrics (energy / CO2 / throughput)")
                    st.caption(
                        f"All time-based quantities use SI seconds via "
                        f"`code_time_to_seconds(t, params)`. Current time unit: "
                        f"**{t_unit}** → factor "
                        f"{_TIME_UNIT_TO_SECONDS.get(t_unit, 1.0):.2e} s.")
                    df_eng = pd.DataFrame([
                        {"Metric": "Bonding time per joint",
                         "Design (nc-Cu)": f"{metrics['t_nc_measured']:.3g} {t_unit} "
                                            f"({metrics['t_nc_s']:.3g} s)",
                         "Reference (Poly-Cu)": f"{metrics['t_poly_measured']:.3g} {t_unit} "
                                            f"({metrics['t_poly_s']:.3g} s)",
                         "Ratio Ref/Design": f"{metrics['S_t']:.1f}x"},
                        {"Metric": "Bonding temperature",
                         "Design (nc-Cu)": f"{T_bond_K-273.15:.0f} C",
                         "Reference (Poly-Cu)": f"{T_bond_K-273.15+metrics['Delta_T_eff']:.0f} C",
                         "Ratio Ref/Design": f"+{metrics['Delta_T_eff']:.0f} C"},
                        {"Metric": "Bonding pressure",
                         "Design (nc-Cu)": f"{metrics['sigma_nc_MPa']:.0f} MPa",
                         "Reference (Poly-Cu)": f"{metrics['sigma_poly_MPa']:.0f} MPa",
                         "Ratio Ref/Design": f"{metrics['S_P']:.1f}x"},
                        {"Metric": "Energy per joint",
                         "Design (nc-Cu)": f"{metrics['E_nc_kWh']:.3g} kWh",
                         "Reference (Poly-Cu)": f"{metrics['E_poly_kWh']:.3g} kWh",
                         "Ratio Ref/Design": f"{metrics['energy_ratio']:.1f}x"},
                        {"Metric": "CO2 per joint",
                         "Design (nc-Cu)": f"{metrics['CO2_nc']:.3g} kg",
                         "Reference (Poly-Cu)": f"{metrics['CO2_poly']:.3g} kg",
                         "Ratio Ref/Design": f"{metrics['co2_ratio']:.1f}x"},
                        {"Metric": "Throughput (joints / hour)",
                         "Design (nc-Cu)": f"{metrics['throughput_nc']:.1f}",
                         "Reference (Poly-Cu)": f"{metrics['throughput_poly']:.1f}",
                         "Ratio Ref/Design": f"{1.0/max(metrics['S_t'],1e-9):.2f}x"},
                        {"Metric": "Au utilization η_Au",
                         "Design (nc-Cu)": f"{metrics['eta_Au_nc']*100:.0f}%",
                         "Reference (Poly-Cu)": f"{metrics['eta_Au_poly']*100:.0f}%",
                         "Ratio Ref/Design": f"{metrics['eta_Au_poly']/max(metrics['eta_Au_nc'],1e-6):.2f}x"},
                    ])
                    st.dataframe(df_eng, use_container_width=True, hide_index=True)

                    # ---- Cost per joint + break-even ----
                    st.markdown("---")
                    st.subheader("Cost per joint (full breakdown)")
                    cn, cp_m = metrics['cost_nc'], metrics['cost_poly']
                    df_cost = pd.DataFrame([
                        {"Line item": "Energy",
                         "Design (nc-Cu) ($)": f"{cn['energy']:.6f}",
                         "Reference (Poly-Cu) ($)": f"{cp_m['energy']:.6f}",
                         "Ref/Design": f"{cp_m['energy']/max(cn['energy'],1e-12):.2f}x"},
                        {"Line item": "Machine (amortised)",
                         "Design (nc-Cu) ($)": f"{cn['machine']:.6f}",
                         "Reference (Poly-Cu) ($)": f"{cp_m['machine']:.6f}",
                         "Ref/Design": f"{cp_m['machine']/max(cn['machine'],1e-12):.2f}x"},
                        {"Line item": "Labor",
                         "Design (nc-Cu) ($)": f"{cn['labor']:.6f}",
                         "Reference (Poly-Cu) ($)": f"{cp_m['labor']:.6f}",
                         "Ref/Design": "-"},
                        {"Line item": "Au material",
                         "Design (nc-Cu) ($)": f"{cn['au_material']:.6f}",
                         "Reference (Poly-Cu) ($)": f"{cp_m['au_material']:.6f}",
                         "Ref/Design": "-"},
                        {"Line item": "Cu material (with nano premium)",
                         "Design (nc-Cu) ($)": f"{cn['cu_material']:.6f}",
                         "Reference (Poly-Cu) ($)": f"{cp_m['cu_material']:.6f}",
                         "Ref/Design": f"{cp_m['cu_material']/max(cn['cu_material'],1e-12):.2f}x"},
                        {"Line item": "TOTAL per joint",
                         "Design (nc-Cu) ($)": f"{cn['total']:.6f}",
                         "Reference (Poly-Cu) ($)": f"{cp_m['total']:.6f}",
                         "Ref/Design": f"{metrics['cost_ratio']:.2f}x"},
                    ])
                    st.dataframe(df_cost, use_container_width=True, hide_index=True)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Cost ratio (Ref/Design)",
                                   f"{metrics['cost_ratio']:.2f}x")
                    with c2:
                        be_prem = metrics['break_even_premium']
                        st.metric("Break-even Cu premium",
                                   f"${be_prem:.2f}/kg" if np.isfinite(be_prem) else "N/A",
                                   help="At what $/kg nano-Cu premium does the energy "
                                        "saving stop justifying the material premium?")
                    with c3:
                        st.metric("Break-even grain size",
                                   f"{metrics['break_even_grain_size']:.0f} nm",
                                   help="Grain size at which S_t drops to 1.5x (the "
                                        "minimum commercially defensible advantage).")

                    # =====================================================
                    # PUBLICATION-QUALITY FIGURES (8)
                    # =====================================================
                    st.markdown("---")
                    st.subheader("Publication-quality figures (8)")
                    st.caption(
                        "Each figure proves a specific claim — see the inline "
                        "captions. All figures use `constrained_layout` and the "
                        "project font fallback (DejaVu Sans + Noto Sans SC). "
                        "Data-driven: S_t is the measured value from the saved "
                        "diagnostics of the two selected simulations.")

                    fig_params = {
                        'D_lattice': params_for_calc.get('D_lattice', 0.01),
                        'D_gb': params_for_calc.get('D_gb', 50.0),
                        'gb_width_nm': gb_width,
                    }

                    # ---- Fig 1: S_t(d) ----
                    with st.expander("Fig 1 — Thermal-budget multiplier S_t(d)",
                                      expanded=True):
                        st.caption(
                            "Proves *why* the nc-Cu choice is on the steep part of "
                            "the curve. Vertical orange line = break-even grain size "
                            f"({metrics['break_even_grain_size']:.0f} nm, where S_t=1.5x). "
                            "Red dot = current design. Blue square = Poly-Cu reference.")
                        fig1 = fig_S_t_vs_d(fig_params, d_current, d_ref)
                        st.pyplot(fig1); plt.close(fig1)

                    # ---- Fig 2: dT Arrhenius ----
                    with st.expander("Fig 2 — Arrhenius temperature penalty ΔT(S_t)",
                                      expanded=False):
                        st.caption(
                            "Exact Arrhenius inversion (black) vs the small-ΔT "
                            "linearisation (blue dashed). Shaded gap = approximation "
                            "error. Red dot = current design point (measured S_t).")
                        fig2 = fig_Delta_T_arrhenius(
                            fig_params, metrics['S_t'], T_K=T_bond_K, Q_gb_kJ=Q_gb_kj)
                        st.pyplot(fig2); plt.close(fig2)

                    # ---- Fig 3: engineering bars ----
                    with st.expander("Fig 3 — Engineering penalty factors",
                                      expanded=False):
                        st.caption(
                            "The headline bar chart — goes in the paper abstract. "
                            "Poly-Cu is uniformly penalised by S_t on time, energy, "
                            "CO2, and Au utilisation; throughput is penalised by 1/S_t. "
                            "S_t here is the MEASURED value from saved diagnostics.")
                        fig3 = fig_engineering_bars(metrics['S_t'])
                        st.pyplot(fig3); plt.close(fig3)

                    # ---- Fig 4: composite MCDA radar ----
                    with st.expander("Fig 4 — Composite MCDA score (geometric mean)",
                                      expanded=False):
                        st.caption(
                            "Single-number answer to 'which configuration is better "
                            "overall?'. The geometric mean is scale-invariant and "
                            "cannot be gamed by one dominant axis — the correct MCDA "
                            "aggregation for ratio metrics.")
                        fig4 = fig_composite_radar(metrics['S_t'])
                        st.pyplot(fig4); plt.close(fig4)

                    # ---- Fig 5: Monte-Carlo dT ----
                    with st.expander("Fig 5 — Monte-Carlo uncertainty on ΔT",
                                      expanded=False):
                        st.caption(
                            f"Pre-empts the reviewer who says 'but you assumed Q'. "
                            f"8000 draws of Q ~ N(Q_gb, σ_Q²), clipped to "
                            f"[0.5, 1.5] * Q_gb. The 95% CI on ΔT is "
                            f"[{metrics['Delta_T_95CI'][0]:+.0f}, "
                            f"{metrics['Delta_T_95CI'][1]:+.0f}] C.")
                        fig5 = fig_montecarlo_DT(
                            metrics['S_t'], T_K=T_bond_K, Q_gb_kJ=Q_gb_kj,
                            sigma_Q_kJ=sigma_Q_kj)
                        st.pyplot(fig5); plt.close(fig5)

                    # ---- Fig 6: measured kinetics ----
                    with st.expander("Fig 6 — Measured bonding kinetics",
                                      expanded=True):
                        st.caption(
                            "MEASURED <c>_Ob(t) for the Design (nc-Cu) and "
                            "Reference (Poly-Cu) simulations, pulled from their "
                            "saved 1D diagnostics. No heuristic — two real "
                            "saved runs, two real curves. Markers show where each "
                            "run first hit the target <c>*.")
                        if t_des.size > 0 and t_ref.size > 0:
                            fig6 = fig_kinetics(
                                t_des, c_des, t_ref, c_ref, target_c,
                                t_unit=t_unit,
                                nc_reached=des_reached,
                                poly_reached=ref_reached)
                            st.pyplot(fig6); plt.close(fig6)
                        else:
                            st.warning("One or both selected runs have no "
                                       "trajectory data in their diagnostics.")

                    # ---- Fig 7: Pareto ----
                    with st.expander("Fig 7 — Pareto: sustainability vs. cost",
                                      expanded=False):
                        st.caption(
                            "Manufacturing reality check. The Pareto sweet spot "
                            "(red star) is where total cost per joint is minimised — "
                            "NOT at the smallest d, because the nano-premium "
                            "outweighs the energy saving below ~40 nm. This is the "
                            "honest engineering answer.")
                        cost_dict = dict(
                            batch_joints=int(batch_joints),
                            furnace_kW=furnace_kW,
                            energy_price_usd_kwh=energy_price,
                            machine_hourly_usd=machine_hourly,
                            cu_mass_kg=cu_mass_kg,
                            bulk_cu_usd_kg=0.0,
                            nano_premium_usd_kg=nano_premium,
                            premium_decay_nm=40.0,
                            premium_ref_nm=15.0,
                        )
                        fig7 = fig_pareto_grain_size(
                            fig_params, metrics['t_nc_s'], d_current, d_ref,
                            cost=cost_dict)
                        st.pyplot(fig7); plt.close(fig7)

                    # ---- Fig 8: sensitivity heatmap ----
                    with st.expander("Fig 8 — Operating window S_t(d, d_ref)",
                                      expanded=False):
                        st.caption(
                            "Design map: every (d, d_ref) pair inside the white "
                            "5x contour gives S_t < 5. The red star is the current "
                            "design point.")
                        fig8 = fig_sensitivity_heatmap(
                            fig_params, d_nc_actual=d_current, d_ref_actual=d_ref)
                        st.pyplot(fig8); plt.close(fig8)

                    # =====================================================
                    # MULTI-RUN PARETO SWEEP (data-driven)
                    # =====================================================
                    st.markdown("---")
                    st.subheader("6. Multi-run Pareto sweep (data-driven)")
                    st.caption(
                        "Select multiple saved runs with varying <d> to build a "
                        "data-driven Pareto front from your actual simulation "
                        "results. Each point is the MEASURED time-to-target for "
                        "that run's grain size — no geometric proxy.")

                    multi_select = st.multiselect(
                        "Select runs for sweep",
                        range(len(sim_options)),
                        format_func=lambda i: sim_options[i]['label'],
                        key="sus_sweep_select")

                    if multi_select:
                        sweep_data = []
                        for idx_sel in multi_select:
                            sd = sim_options[idx_sel]
                            diag = sd['sim'].get('diagnostics', {})
                            t_sw = np.array(diag.get('time', []))
                            c_sw = np.array(diag.get('c_bot', []))
                            t_r, reached = _time_to_target(t_sw, c_sw, target_c)
                            sweep_data.append({
                                'd (nm)': sd['d'],
                                'Type': classify_substrate(sd['d']),
                                't_reach': (float(t_r) if np.isfinite(t_r)
                                            else float('nan')),
                                'Reached Target': reached,
                                'sim_id': sd['id'],
                            })

                        if sweep_data:
                            df_sweep = pd.DataFrame(sweep_data).sort_values('d (nm)')
                            st.dataframe(df_sweep, use_container_width=True,
                                         hide_index=True)

                            # Plot data-driven t_reach vs d
                            df_plot = df_sweep[df_sweep['Reached Target']].copy()
                            if not df_plot.empty:
                                fig, ax = plt.subplots(figsize=(9, 5))
                                ax.loglog(df_plot['d (nm)'], df_plot['t_reach'],
                                          'o-', color='darkgreen', lw=2, ms=9,
                                          label='Measured t_reach')
                                ax.set_xlabel("Average grain size <d> (nm)")
                                ax.set_ylabel(f"Time to reach target ({t_unit})")
                                ax.axvline(100, color='gray', ls='--', alpha=0.5,
                                           label='nc / Poly threshold (100 nm)')
                                ax.legend()
                                ax.grid(True, which='both', alpha=0.3)
                                ax.set_title("Data-driven t_reach(d) from saved simulations")
                                st.pyplot(fig); plt.close(fig)
                            else:
                                st.info("No selected runs reached the target "
                                        f"<c>*={target_c:.3f}. Lower the target "
                                        "threshold or increase n_steps.")
                    else:
                        st.caption("Select one or more runs above to build the "
                                   "t_reach(d) sweep plot.")

                    # =====================================================
                    # PAPER-READY SUMMARY
                    # =====================================================
                    st.markdown("---")
                    st.subheader("Paper-ready summary")
                    s_meas_str = (f"measured S_t = {metrics['S_t_measured']:.2f}"
                                   if np.isfinite(metrics['S_t_measured'])
                                   else f"geometric S_t = {metrics['S_t']:.2f}")
                    dT_med = metrics['Delta_T_median']
                    dT_lo, dT_hi = metrics['Delta_T_95CI']
                    dT_ci_str = (f" (95% CI: [{dT_lo:+.0f}, {dT_hi:+.0f}] C, "
                                  f"σ_Q = {sigma_Q_kj:.0f} kJ/mol)"
                                  if np.isfinite(dT_med) and dT_med > 0 else "")
                    summary_text = (
                        f"*To quantify the sustainability advantage of the nc-Cu "
                        f"design, we compared two saved phase-field runs — a "
                        f"{d_current:.0f} nm nc-Cu design and a {d_ref:.0f} nm "
                        f"Poly-Cu reference — using their measured 1D diagnostics "
                        f"(time-series of region-averaged Au concentration). Because "
                        f"the Hart effective diffusivity scales linearly with the "
                        f"grain-boundary volume fraction φ_gb ≈ 1/d, replacing "
                        f"the {d_current:.0f} nm nc-Cu layer with {d_ref:.0f} nm Poly-Cu "
                        f"reduces D_eff by a factor of {s_meas_str}. To reach the "
                        f"same region-averaged Au concentration (<c>_Ob = {target_c:.2f}, "
                        f"corresponding to {metrics['eta_Au_nc']*100:.0f}% utilization "
                        f"of the Au inventory), the Poly-Cu reference requires either "
                        f"(i) a {metrics['S_t']:.1f}x longer bonding time at "
                        f"{T_bond_K-273.15:.0f} C, (ii) a +{metrics['Delta_T_eff']:.0f} C "
                        f"temperature increase at fixed throughput (using "
                        f"Q_eff = {metrics['Q_eff_poly_kJmol']:.1f} kJ/mol, the "
                        f"D-weighted average of Q_gb and Q_lat){dT_ci_str}, or "
                        f"(iii) a {metrics['S_P']:.1f}x higher bonding pressure "
                        f"(~{metrics['sigma_poly_MPa']:.0f} MPa, impractical for "
                        f"power modules). The composite MCDA score is "
                        f"Σ = {metrics['composite_score']:.2f} (geometric mean over "
                        f"energy, CO2, throughput, Au utilisation, and cost axes). "
                        f"The nc-Cu design therefore delivers a "
                        f"{metrics['S_t']:.1f}x reduction in thermal energy per joint "
                        f"({metrics['E_nc_kWh']:.2e} vs {metrics['E_poly_kWh']:.2e} kWh), "
                        f"a {metrics['S_t']:.1f}x throughput gain, and a "
                        f"{metrics['eta_Au_nc']/max(metrics['eta_Au_poly'],1e-6):.1f}x "
                        f"improvement in precious-metal utilization. The cost-per-joint "
                        f"break-even Cu premium is "
                        f"${metrics['break_even_premium']:.2f}/kg — well above the "
                        f"current market gap between nc-Cu and bulk Cu.*"
                    )
                    st.markdown(summary_text)

                    # =====================================================
                    # EXPORT (JSON + Markdown)
                    # =====================================================
                    st.markdown("---")
                    st.subheader("Export sustainability report")
                    rep = build_sustainability_report(metrics, params_for_calc,
                                                       summary_text=summary_text)
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        json_bytes = json.dumps(rep, indent=2, cls=NumpyEncoder,
                                                  default=str).encode()
                        st.download_button(
                            "JSON report",
                            BytesIO(json_bytes),
                            file_name=(f"sustainability_{rep['params_hash']}.json"),
                            mime="application/json",
                            help="Full machine-readable report including all metrics, "
                                 "95% CI, cost breakdown, and the parameter set.")
                    with ec2:
                        md_text = make_markdown_report(rep)
                        st.download_button(
                            "Markdown report",
                            md_text,
                            file_name=(f"sustainability_{rep['params_hash']}.md"),
                            mime="text/markdown",
                            help="Human-readable summary in Markdown — drop straight "
                                 "into a paper appendix or repo README.")

                else:
                    # Cannot compute measured S_t
                    st.warning(
                        f"⚠️ Cannot compute measured S_t — one or both "
                        f"selected runs did not reach the target "
                        f"<c>*={target_c:.3f} within their saved trajectories. "
                        f"Design reached: {des_reached}. Reference reached: "
                        f"{ref_reached}. To fix: increase n_steps, lower the target "
                        f"threshold, or select simulations that did reach the target.")

                    # Still show what we have
                    if t_des.size > 0:
                        st.caption(f"Design max <c>_Ob = {float(np.max(c_des)):.4f} "
                                   f"(target {target_c:.3f})")
                    if t_ref.size > 0:
                        st.caption(f"Reference max <c>_Ob = {float(np.max(c_ref)):.4f} "
                                   f"(target {target_c:.3f})")

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

            safe_params = {}
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, (np.integer, np.floating, np.bool_)):
                    v = v.item()
                safe_params[k] = v
            st.json(safe_params)

            if history:
                idx = st.slider("Frame", 0, len(history) - 1,
                                len(history) - 1,
                                key=f"view_frame_{sid}")
                vis = EnhancedTwinVisualizer(Nx, Ny, dx, dy,
                                              dt=params['dt'])
                style = {
                    'colormaps': {f: fset.get('cmap', FIELD_DEFAULT_CMAPS.get(f, 'viridis'))
                                   for f, fset in viz_cfg['field_settings'].items()},
                    'field_settings': viz_cfg['field_settings'],
                    'title_font_size': viz_cfg['title_font_size'],
                    'label_font_size': viz_cfg['label_font_size'],
                    'colorbar_fraction': viz_cfg['colorbar_fraction'],
                    'colorbar_pad': viz_cfg['colorbar_pad'],
                    'figure_customization': viz_cfg.get('figure_customization', {}),
                }
                fig = vis.create_multi_field_comparison(history[idx], style)
                if fig is not None:
                    st.pyplot(fig); plt.close(fig)

            if st.button("🗑️ Delete"):
                SimulationDatabase.delete(sid)
                if 'view_id' in st.session_state:
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

            # v3.4.17: Mode selector — Spatial (last frame) OR Temporal (kinetics)
            cmp_mode = st.radio(
                "Comparison mode",
                ["Spatial (Last Frame)", "Temporal (Kinetics)"],
                horizontal=True,
                key="cmp_mode",
                help=("Spatial: side-by-side 2D field maps of the last saved "
                      "frame. Temporal: overlay <c>_Ob(t) curves from the saved "
                      "1D diagnostics, with time-to-target markers — the "
                      "data-driven way to compare bonding kinetics across "
                      "grain sizes."))

            if cmp_mode == "Spatial (Last Frame)":
                field = st.selectbox("Field",
                                      list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=0, key="cmp_field")
                fs_cmp = viz_cfg['field_settings'].get(field, {})
                cmap_name = fs_cmp.get('cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                cmap = ColormapRegistry.resolve(cmap_name,
                                                 invert=fs_cmp.get('invert', False))

                # v3.3: unified bounds for the comparison grid
                all_vals = []
                for sim in sims:
                    h = sim['results_history']
                    if h and field in h[-1]:
                        dd, _ = _rescale_field(field, h[-1][field])
                        finite = dd[np.isfinite(dd)]
                        if finite.size:
                            all_vals.append(finite)

                if not fs_cmp.get('auto', True):
                    # User-provided override — respect it exactly
                    drng = FIELD_DEFAULT_RNG.get(field, (0.0, 1.0))
                    vmin_g = (fs_cmp.get('vmin') if fs_cmp.get('vmin') is not None
                              else drng[0])
                    vmax_g = (fs_cmp.get('vmax') if fs_cmp.get('vmax') is not None
                              else drng[1])
                elif field in FIELD_DEFAULT_RNG:
                    vmin_g, vmax_g = FIELD_DEFAULT_RNG[field]
                elif all_vals:
                    stack = np.concatenate(all_vals)
                    vmin_g = float(np.percentile(stack, 2))
                    vmax_g = float(np.percentile(stack, 98))
                    if field == 'sigma_h':
                        zm = max(abs(vmin_g), abs(vmax_g))
                        vmin_g, vmax_g = -zm, zm
                else:
                    vmin_g, vmax_g = None, None

                cols = min(3, len(sims))
                rows = (len(sims) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                axes = np.array(axes).flatten()

                for i, sim in enumerate(sims):
                    p = sim['params']
                    h = sim['results_history']
                    if not h:
                        continue
                    last = h[-1]
                    if field not in last:
                        continue
                    d, unit = _rescale_field(field, last[field])
                    extent = [-p['Nx']*p['dx']/2, p['Nx']*p['dx']/2,
                              -p['Ny']*p['dy']/2, p['Ny']*p['dy']/2]
                    ax = axes[i]
                    interp = 'nearest' if field in DISCRETE_FIELDS else 'bilinear'
                    im = ax.imshow(d, extent=extent, origin='lower', cmap=cmap,
                                   vmin=vmin_g, vmax=vmax_g,
                                   interpolation=interp, aspect='equal')
                    ax.set_title(f"{sim['id']} — {field}")
                    if field != 'grain_map':
                        plt.colorbar(im, ax=ax, fraction=0.046)
                for k in range(len(sims), len(axes)):
                    axes[k].axis('off')
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

            else:  # Temporal (Kinetics)
                st.subheader("Temporal evolution: <c>_Ob(t)")
                st.caption(
                    "Comparing bonding kinetics across different grain sizes. "
                    "Each curve is the region-averaged Au concentration in the "
                    "bottom layer, pulled from the saved 1D diagnostics "
                    "(`solver.history`). Substrate type is auto-classified: "
                    "d < 100 nm = nc-Cu; d >= 100 nm = Poly-Cu. Dashed line "
                    "= target threshold; circles mark time-to-target.")

                target_c_cmp = st.slider(
                    "Target <c>*", 0.01, 0.20, 0.06, 0.005,
                    key="cmp_target",
                    help="Region-averaged Au concentration threshold. The "
                         "time-to-target marker shows when each run first "
                         "crossed this value.")

                # Check which sims have diagnostics
                sims_with_diag = [(i, s) for i, s in enumerate(sims)
                                   if s.get('diagnostics')]
                if not sims_with_diag:
                    st.warning(
                        "⚠️ None of the selected simulations have "
                        "saved 1D diagnostics. This happens for runs saved "
                        "before v3.4.17. Re-run and save to populate the "
                        "diagnostics, then return here to see the temporal "
                        "comparison.")
                else:
                    missing = len(sims) - len(sims_with_diag)
                    if missing > 0:
                        st.caption(
                            f"Note: {missing} of {len(sims)} selected runs "
                            f"have no saved diagnostics and are omitted from "
                            f"the temporal plot.")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sims_with_diag), 1)))

                    any_plotted = False
                    t_unit_cmp = ""
                    for k, (orig_idx, sim) in enumerate(sims_with_diag):
                        diag = sim.get('diagnostics', {})
                        t = np.array(diag.get('time', []))
                        c_bot = np.array(diag.get('c_bot', []))
                        if t.size == 0 or c_bot.size == 0:
                            continue
                        any_plotted = True
                        d_val = float(sim['params'].get('nc_grain_size_nm', 0))
                        sub_type = classify_substrate(d_val)
                        t_unit_cmp = get_time_unit_label(sim['params'])
                        label = f"d={d_val:.0f} nm ({sub_type})"
                        ax.plot(t, c_bot, color=colors[k], lw=2.5, label=label)
                        # Mark time-to-target
                        t_cross, reached = _time_to_target(t, c_bot, target_c_cmp)
                        if reached:
                            ax.plot(t_cross, target_c_cmp, 'o',
                                    color=colors[k], ms=10, zorder=5)
                            ax.annotate(
                                f"t={t_cross:.2f}",
                                (t_cross, target_c_cmp),
                                textcoords="offset points", xytext=(0, 10),
                                ha='center', fontsize=8, color=colors[k])

                    if any_plotted:
                        ax.axhline(target_c_cmp, color='k', ls='--', lw=1.5,
                                   label=f'Target {target_c_cmp:.3f}')
                        ax.set_xlabel(f"Time ({t_unit_cmp})")
                        ax.set_ylabel("Region-averaged <c>_Ob (bottom layer)")
                        ax.set_title("Bonding kinetics across grain sizes "
                                     "(data-driven)")
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig); plt.close(fig)
                    else:
                        plt.close(fig)
                        st.info("No trajectory data could be extracted from "
                                "the selected runs.")


if __name__ == "__main__":
    main()
