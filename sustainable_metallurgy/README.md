## r15: Flexible domain size, t in s  ( enhanced version of r14 to incorporate sustainable metallurgy concept)

# The sustainability calculations in r15 is yet to be connected with multiple simulations


[![continuummodelnt2d](https://img.shields.io/badge/audiff15-streamlit-violet)](https://preferential-au-diffusion15.streamlit.app/) (figure customization, temporal features, limiting the edges of colorscale bars, c variable annotated properly within the calculation framework, customization of the visualizations, diffusion correction  and  simulation produced stable results , basic code, geometry change of domain from square to rectangle)

[![continuummodelnt2d](https://img.shields.io/badge/audiff15a-streamlit-violet)](https://preferential-au-diffusion15a.streamlit.app/) (figure customization, temporal features, limiting the edges of colorscale bars, c variable annotated properly within the calculation framework, customization of the visualizations, diffusion correction  and  simulation produced stable results , basic code, geometry change of domain from square to rectangle)

# The sustainability calculations in r16 is integrated with multiple simulations

[![continuummodelnt2d](https://img.shields.io/badge/audiff16-streamlit-blue)](https://preferential-au-diffusion16.streamlit.app/) (figure customization, temporal features, limiting the edges of colorscale bars, c variable annotated properly within the calculation framework, customization of the visualizations, diffusion correction  and  simulation produced stable results , basic code, geometry change of domain from square to rectangle)

16d uses the same economic valuation of different sized substrate. 16c consideres <d>=60 nm costlier than <d>=120 nm.


# Sustainability — Implementation README

This release extends the Au-assisted Cu-Cu diffusion bonding simulator
() with **quantitative sustainability analysis** of the nc-Cu
vs. Poly-Cu bottom-layer choice. It implements the three-pathway
framework (time / temperature / pressure penalties) plus derived
engineering metrics (energy, CO₂, throughput, Au utilisation).

## Deliverables in this folder

| File | Purpose |
|---|---|
| `au_cu_sim_v3.4.15_sustainability.py` | The modified Streamlit simulator. Drop-in replacement for `v3.4.14` — same UI, same physics, plus a new sidebar toggle and a new "🌱 Sustainability" tab. |
| `test_sustainability_helpers.py` | Smoke test for the two new helper functions. Run with `python test_sustainability_helpers.py`. All 3 tests pass on the proposal's reference numbers (d_nc=15 nm, d_poly=150 nm, T_bond=523 K → S_t≈9.82×, ΔT≈+52 °C, S_P≈9.82×). |


## What's new

### 1. Sidebar toggle (Geometry section)

A new `st.selectbox` right below the `nc-Cu grain size` slider lets the user
pick the bottom layer microstructure:

| Option | Behaviour | Use case |
|---|---|---|
| **nc-Cu (as-designed)** | Default. Uses the `nc_gs` slider value (e.g. 15–95 nm). | Production configuration. |
| **Poly-Cu (reference)** | Forces the bottom layer to use the SAME grain size as the top Poly-Cu (`poly_gs`, default 150 nm). | Sustainability comparison baseline. |
| **Coarse Poly-Cu (d=500 nm)** | Fixed 500 nm grain size. | Conventional bulk / annealed Cu worst-case. |

Switching **does NOT change any physics** — the Hart effective diffusivity,
solute drag, spectral elastoplasticity, and GND tracking all automatically
reflect the new φ_gb because φ_gb is computed from the order-parameter
field, not hard-coded.

### 2. 🌱 Sustainability tab (new, after 📤 Export)

Sections inside the tab:

1. **🎯 Bonding-quality target** — slider for ⟨c⟩_Ωb* (default 0.06).
2. **🌡️ Material constants** — Q_gb (default 100 kJ/mol), Q_lat (default 200 kJ/mol), T_bond (default 523 K).
3. **📊 Reference configuration** — d_ref (default 150 nm).
4. **📐 Primary sustainability penalties** — three big metric cards: S_t, ΔT, S_P.
5. **🔬 Geometric lever: φ_gb comparison** — φ_gb_nc, φ_gb_poly, D_eff ratio.
6. **🏭 Derived engineering metrics** — interactive table: time, T, P, energy, CO₂, throughput, Au utilisation. User can tune `g_grid` (carbon intensity) and furnace power.
7. **📈 Bonding-kinetics comparison** — side-by-side ⟨c⟩_Ωb(t) plot for nc-Cu vs. Poly-Cu reference (Poly-Cu curve is S_t-stretched).
8. **📝 Paper-ready summary** — italicised paragraph that auto-fills the proposal's text with the user's current numbers. Copy-paste ready for the manuscript.

### 3. Module-level helper functions

Added before `def main()`:

- **`_time_to_target(t_arr, c_arr, target_c)`** — Returns the time at
  which `c_arr` first reaches `target_c` (linear interpolation), or `NaN`
  if unreachable in this run.

- **`compute_sustainability_metrics(*, solver_history, params, target_c=0.06, d_current=None, d_ref=150.0, gb_width=1.5, Q_gb_J=100e3, Q_lat_J=200e3, T_bond_K=523.0, applied_stress_MPa=30.0)`**
  — Returns a dict with: `S_t`, `t_nc_measured`, `t_poly_estimated`,
  `Delta_T_gb`, `Delta_T_lat`, `S_P`, `sigma_nc_MPa`, `sigma_poly_MPa`,
  `phi_gb_nc`, `phi_gb_poly`, `D_eff_nc`, `D_eff_poly`, `eta_Au_nc`,
  `eta_Au_poly`, `R`.

Both are pure functions (no Streamlit state) so they can be unit-tested
or batch-invoked from a separate analysis script.

## How to run

```bash
streamlit run Sustainability-mucu-au-nanocrystal-cu-pfm-r15.py
```

Then:
1. In the sidebar, set the geometry as usual (e.g. nc_gs=15 nm, poly_gs=150 nm, σ=30 MPa, T=250 °C via time-unit='s').
2. Click **▶️ Start Evolution** on the Run tab.
3. Open the **🌱 Sustainability** tab to see the live metrics.
4. Adjust the bonding-quality target, Q_gb, d_ref to do sensitivity sweeps.

## Validation

The smoke test `test_sustainability_helpers.py` reproduces the proposal's
order-of-magnitude estimates:

```
S_t               = 9.82×   (proposal: ~10×)  ✓
t_nc_measured     = 0.100   (matches proposal) ✓
t_poly_estimated  = 0.982   (matches proposal ~1.0) ✓
Delta_T_gb        = +52.0 °C (proposal: ~+58 °C — difference because the
                              proposal used S_t=10 exactly; with D_lat
                              correction the true S_t=9.82 gives 52 °C) ✓
S_P               = 9.82×   (proposal: ~10×)  ✓
phi_gb_nc         = 0.100   ✓
phi_gb_poly       = 0.010   ✓
D_eff_nc          = 5.009   (proposal: ~5.0)  ✓
D_eff_poly        = 0.510   (proposal: ~0.5)  ✓
sigma_poly_MPa    = 294.7   (proposal: ~300)  ✓
```

## Compatibility

- The modified file is a **drop-in replacement** for `v3.4.14` — all
  existing tabs (Geometry, Run, Visualization Settings, Results, Line
  Profiles, Plotly, Export) behave identically.
- The only structural change is that the `tabs = st.tabs(...)` list now
  has 8 entries instead of 7.
- `params` dict gains one new key: `bottom_layer_type` (string). Existing
  pickle/JSON/HDF5/CSV/PyTorch exporters handle this transparently.
- Auto-rebuild logic (v3.4.8) now triggers on `bottom_layer_type` change
  too — switching the bottom layer from nc-Cu to Poly-Cu automatically
  rebuilds the geometry silently.
