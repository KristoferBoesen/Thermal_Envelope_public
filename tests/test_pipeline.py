"""Unit tests for the milestone root-finding functions.

Uses inline properties and config dicts to avoid any dependency on
solver_config.yaml, following the _make_sim() pattern in test_fem_solver.py.

Analytical basis
----------------
For a slowly-decaying source (λ → 0), the FEM solution reaches quasi-steady
state much faster than the source changes.  The steady-state centreline
temperature is:

    T_center = T_inf + Q_vol × (R/(2h) + R²/(4k))

Inverting for h at the temperature limit gives an analytical h_min that can
be compared to the optimiser output (within ±15% to account for transient
overshoot).

For find_total_decay_years with a single-term decay
Q_vol(t) = A·exp(−λt)·ρ_eff·loading, the time at which Q_vol = Q_allowable is:

    t = (1/λ) × ln(A·ρ_eff·loading / Q_allowable)
"""

import math
import numpy as np
import pytest

from aethon.analysis.pipeline import find_min_h_active, find_total_decay_years
from aethon.constants import KELVIN_OFFSET

# ---------------------------------------------------------------------------
# Shared constants for all pipeline tests
# ---------------------------------------------------------------------------
_K   = 2.0      # W/(m·K)   — constant thermal conductivity
_CP  = 500.0    # J/(kg·K)  — constant specific heat
_RHO = 2500.0   # kg/m³     — base matrix density


def _make_props(decay_func=None):
    """Return a minimal properties dict with constant k and cp."""
    return {
        "rho_base": _RHO,
        "decay":    decay_func or (lambda t: 100.0 * np.exp(-0.5 * t)),
        "cp":       lambda T: np.full_like(np.asarray(T, dtype=float), _CP),
        "k":        lambda T: np.full_like(np.asarray(T, dtype=float), _K),
    }


def _make_cfg(
    centerline_C=400.0,
    sf=1.0,
    nodes=15,
    passive_ambient_C=40.0,
    passive_h=5.0,
):
    """Return a minimal config dict for pipeline tests."""
    return {
        "centerline_limit_C": centerline_C,
        "safety_factor":      sf,
        "passive_ambient_C":  passive_ambient_C,
        "passive_h":          passive_h,
        "h_passive":          passive_h,
        "nodes":              nodes,
        "max_years":          50.0,
    }


# ===========================================================================
# Tests for find_min_h_active
# ===========================================================================

class TestFindMinHActive:

    def test_passive_sufficient_returns_nan(self):
        """Very low decay → peak T well below limit at h_passive → returns nan."""
        props = _make_props(decay_func=lambda t: 0.001 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        result = find_min_h_active(
            R=0.1,
            loading_fraction=0.05,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
            ambient_C=40.0,
            cooling_years=0.0,
        )
        assert np.isnan(result), f"Expected nan, got {result}"

    def test_infeasible_returns_inf(self):
        """Enormous source, large radius → infeasible at H_SEARCH_MAX → returns inf."""
        props = _make_props(decay_func=lambda t: 1000.0 * np.exp(-0.001 * t))
        cfg   = _make_cfg()
        result = find_min_h_active(
            R=0.5,
            loading_fraction=0.3,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
            ambient_C=40.0,
            cooling_years=0.0,
        )
        assert np.isinf(result) and result > 0, f"Expected +inf, got {result}"

    def test_min_h_quasi_steady_state(self):
        """
        Slow decay (λ = 0.001 yr⁻¹) → analytical quasi-steady-state h_min.

        Parameters:
          A = 500 W/kg, λ = 0.001 yr⁻¹, loading = 0.15, ρ_base = 2500 kg/m³
          → ρ_eff = 2941.2 kg/m³, Q_vol ≈ 220 588 W/m³
          R = 0.1 m, k = 2.0, T_lim = 673.15 K, T_inf = 313.15 K

        Analytical:
          Δ = (T_lim − T_inf) / Q_vol − R²/(4k) = 360/220588 − 0.00125
              = 0.001632 − 0.00125 = 0.000382
          h_min = R / (2 × 0.000382) ≈ 131 W/(m²·K)

        Tolerance ±15% to account for transient overshoot.
        """
        props = _make_props(decay_func=lambda t: 500.0 * np.exp(-0.001 * t))
        cfg   = _make_cfg(nodes=20)
        result = find_min_h_active(
            R=0.1,
            loading_fraction=0.15,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
            ambient_C=40.0,
            cooling_years=0.0,
        )
        analytical_h = 131.0
        assert np.isfinite(result), f"Expected finite h_min, got {result}"
        assert 0.85 * analytical_h <= result <= 1.15 * analytical_h, (
            f"h_min = {result:.2f} outside ±15% of analytical {analytical_h}"
        )

    def test_min_h_increases_with_loading(self):
        """Higher waste loading → higher Q_vol → requires higher h_min."""
        props = _make_props(decay_func=lambda t: 500.0 * np.exp(-0.001 * t))
        cfg   = _make_cfg()
        h_results = [
            find_min_h_active(
                R=0.1,
                loading_fraction=loading,
                properties=props,
                rho_base=_RHO,
                cfg=cfg,
                ambient_C=40.0,
                cooling_years=0.0,
            )
            for loading in (0.05, 0.10, 0.15)
        ]

        assert all(np.isfinite(h) for h in h_results), (
            f"All h_min should be finite: {h_results}"
        )
        assert h_results[0] < h_results[1] < h_results[2], (
            f"h_min should increase with loading: {h_results}"
        )

    def test_more_pre_cooling_lowers_required_h(self):
        """
        Letting the waste decay before encapsulation reduces the cooling duty.

        This is the trade the archetype library exposes: waiting longer buys
        the same margin a stronger cooling system would.
        """
        props = _make_props(decay_func=lambda t: 500.0 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        early = find_min_h_active(
            R=0.1, loading_fraction=0.15, properties=props, rho_base=_RHO,
            cfg=cfg, ambient_C=40.0, cooling_years=0.0,
        )
        late = find_min_h_active(
            R=0.1, loading_fraction=0.15, properties=props, rho_base=_RHO,
            cfg=cfg, ambient_C=40.0, cooling_years=3.0,
        )
        assert np.isfinite(early)
        assert (np.isnan(late) or late < early), (
            f"h_min should fall with pre-cooling: {early} -> {late}"
        )

    def test_cooler_facility_lowers_required_h(self):
        """A colder hall needs less convective performance for the same design."""
        props = _make_props(decay_func=lambda t: 500.0 * np.exp(-0.001 * t))
        cfg   = _make_cfg()
        warm = find_min_h_active(
            R=0.1, loading_fraction=0.15, properties=props, rho_base=_RHO,
            cfg=cfg, ambient_C=40.0, cooling_years=0.0,
        )
        cold = find_min_h_active(
            R=0.1, loading_fraction=0.15, properties=props, rho_base=_RHO,
            cfg=cfg, ambient_C=15.0, cooling_years=0.0,
        )
        assert cold < warm, f"colder ambient should need less h: {warm} -> {cold}"


# ===========================================================================
# Tests for find_total_decay_years
# ===========================================================================

_BENTONITE_SURFACE_C = 100.0


class TestFindTotalDecayYears:

    def test_immediately_safe_returns_zero(self):
        """Tiny source → Q_vol < Q_allowable at t=0 → returns 0.0."""
        props = _make_props(decay_func=lambda t: 0.001 * np.exp(-1.0 * t))
        cfg   = _make_cfg()
        result = find_total_decay_years(
            R=0.3,
            loading_fraction=0.05,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
            surface_limit_C=_BENTONITE_SURFACE_C,
        )
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_never_safe_returns_inf(self):
        """Near-constant source: Q_vol > Q_allowable even at t = 1000 yr → inf."""
        props = _make_props(decay_func=lambda t: 100.0 * np.exp(-1e-5 * t))
        cfg   = _make_cfg()
        result = find_total_decay_years(
            R=0.3,
            loading_fraction=0.3,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
            surface_limit_C=_BENTONITE_SURFACE_C,
        )
        assert np.isinf(result) and result > 0, f"Expected +inf, got {result}"

    def test_closed_form(self):
        """
        Single-term decay → closed-form crossing time.

        Parameters:
          A = 50 W/kg, λ = 0.5 yr⁻¹, loading = 0.1, ρ_base = 2500 kg/m³
          → ρ_eff = 2777.8 kg/m³,  Q_vol(t) = 13888.9 × exp(−0.5t)  [W/m³]
          R = 0.3 m, surface limit 100 °C, h_passive = 5, k = 2.0

        Q_allowable (surface-limited):
          Q_surf = (373.15 − 313.15) / (0.3 / 10) = 2000 W/m³

        Closed form:
          t = (1/λ) × ln(A·ρ_eff·loading / Q_allow)
            = 2 × ln(13888.9 / 2000) ≈ 3.876 yr
        """
        props = _make_props(decay_func=lambda t: 50.0 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        result = find_total_decay_years(
            R=0.3,
            loading_fraction=0.1,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
            surface_limit_C=_BENTONITE_SURFACE_C,
        )

        lam      = 0.5
        A        = 50.0
        eff_rho  = _RHO / (1.0 - 0.1)   # 2777.8
        loading  = 0.1
        T_inf_K  = 40.0 + KELVIN_OFFSET
        T_surf_K = _BENTONITE_SURFACE_C + KELVIN_OFFSET
        Q_allow  = (T_surf_K - T_inf_K) / (0.3 / (2.0 * 5.0))   # = 2000.0
        t_analytical = (1.0 / lam) * math.log(A * eff_rho * loading / Q_allow)

        assert np.isfinite(result), f"Expected finite t, got {result}"
        assert abs(result - t_analytical) <= 0.1, (
            f"t = {result:.4f} yr, analytical = {t_analytical:.4f} yr "
            f"(diff = {abs(result - t_analytical):.4f} yr)"
        )

    def test_increases_with_loading(self):
        """Higher waste loading → longer wait before passive safety."""
        props = _make_props(decay_func=lambda t: 50.0 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        t_results = [
            find_total_decay_years(
                R=0.3,
                loading_fraction=loading,
                properties=props,
                rho_base=_RHO,
                cfg=cfg,
                surface_limit_C=_BENTONITE_SURFACE_C,
            )
            for loading in (0.05, 0.10)
        ]

        assert all(np.isfinite(t) for t in t_results), (
            f"Both times should be finite: {t_results}"
        )
        assert t_results[0] < t_results[1], (
            f"Time should increase with loading: {t_results}"
        )

    def test_dropping_surface_limit_is_never_later(self):
        """
        Removing the surface constraint can only raise the allowable heat,
        so the centreline-only milestone cannot come later.  This is what
        guarantees t_coolers_off <= t_geo.
        """
        props = _make_props(decay_func=lambda t: 50.0 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        kwargs = dict(
            R=0.3, loading_fraction=0.1, properties=props,
            rho_base=_RHO, cfg=cfg,
        )
        with_surface = find_total_decay_years(
            surface_limit_C=_BENTONITE_SURFACE_C, **kwargs,
        )
        centre_only = find_total_decay_years(surface_limit_C=np.inf, **kwargs)

        assert centre_only <= with_surface + 1e-9
