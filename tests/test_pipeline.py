"""Unit tests for the design-envelope pipeline optimisation loops.

Uses inline properties and config dicts to avoid any dependency on solver_config.yaml,
following the _make_sim() pattern in test_fem_solver.py.

Analytical basis
----------------
For a slowly-decaying source (λ → 0), the FEM solution reaches quasi-steady
state much faster than the source changes.  The steady-state centreline
temperature is:

    T_center = T_inf + Q_vol × (R/(2h) + R²/(4k))

Inverting for h at the temperature limit gives an analytical h_min that can
be compared to the optimiser output (within ±15% to account for transient
overshoot).

For find_min_cooling_years with a single-term decay
Q_vol(t) = A·exp(−λt)·ρ_eff·loading, the time at which Q_vol = Q_allowable
is:

    t_min = (1/λ) × ln(A·ρ_eff·loading / Q_allowable)
"""

import math
import numpy as np
import pytest

from thermal_envelope.analysis.pipeline import find_min_h_active, find_min_cooling_years
from thermal_envelope.constants import KELVIN_OFFSET

# ---------------------------------------------------------------------------
# Shared constants for all pipeline tests
# ---------------------------------------------------------------------------
_K   = 2.0      # W/(m·K)   — constant thermal conductivity
_CP  = 500.0    # J/(kg·K)  — constant specific heat
_RHO = 2500.0   # kg/m³     — base glass density


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
    surface_C=100.0,
    sf=1.0,
    nodes=15,
    cooling_months=0.0,
    ambient_C=40.0,
    h_passive=5.0,
):
    """Return a minimal config dict for pipeline tests."""
    return {
        "centerline_limit_C": centerline_C,
        "safety_factor":      sf,
        "ambient_temp_C":     ambient_C,
        "h_passive":          h_passive,
        "surface_limits_C":   {"Bentonite": surface_C, "Salt": 200.0},
        "cooling_months":     cooling_months,
        "nodes":              nodes,
        "max_years":          50.0,
    }


# ===========================================================================
# Tests for find_min_h_active
# ===========================================================================

class TestFindMinHActive:

    def test_passive_sufficient_returns_nan(self):
        """Very low decay → peak T well below limit at h_passive → returns nan."""
        # Q_vol ≈ 0.001 × 2631 × 0.05 ≈ 0.13 W/m³ → ΔT ≈ 0.001 K
        props = _make_props(decay_func=lambda t: 0.001 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        result = find_min_h_active(
            R=0.1,
            loading_fraction=0.05,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
        )
        assert np.isnan(result), f"Expected nan, got {result}"

    def test_infeasible_returns_inf(self):
        """Enormous source, large radius → infeasible at H_SEARCH_MAX → returns inf."""
        # Q_vol ≈ 1000 × 3571 × 0.3 ≈ 1.07e6 W/m³ → ΔT at h=2000 >> 360 K
        props = _make_props(decay_func=lambda t: 1000.0 * np.exp(-0.001 * t))
        cfg   = _make_cfg()
        result = find_min_h_active(
            R=0.5,
            loading_fraction=0.3,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
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
        )
        analytical_h = 131.0
        assert np.isfinite(result), f"Expected finite h_min, got {result}"
        assert 0.85 * analytical_h <= result <= 1.15 * analytical_h, (
            f"h_min = {result:.2f} outside ±15% of analytical {analytical_h}"
        )

    def test_min_h_increases_with_loading(self):
        """Higher waste loading → higher Q_vol → requires higher h_min."""
        # Moderate decay so all loadings need active cooling
        props = _make_props(decay_func=lambda t: 500.0 * np.exp(-0.001 * t))
        cfg   = _make_cfg(nodes=15)
        h_results = []
        for loading in [0.05, 0.10, 0.15]:
            h = find_min_h_active(
                R=0.1,
                loading_fraction=loading,
                properties=props,
                rho_base=_RHO,
                cfg=cfg,
            )
            h_results.append(h)

        # All should be finite (passive not enough for any of them)
        assert all(np.isfinite(h) for h in h_results), (
            f"All h_min should be finite: {h_results}"
        )
        # Strict monotone increase
        assert h_results[0] < h_results[1] < h_results[2], (
            f"h_min should increase with loading: {h_results}"
        )


# ===========================================================================
# Tests for find_min_cooling_years
# ===========================================================================

class TestFindMinCoolingYears:

    def test_cooling_immediately_safe_returns_zero(self):
        """Tiny source → Q_vol < Q_allowable at t=0 → returns 0.0."""
        # Q_vol ≈ 0.001 × 2631 × 0.05 ≈ 0.13 W/m³ << Q_allowable ≈ 2000 W/m³
        props = _make_props(decay_func=lambda t: 0.001 * np.exp(-1.0 * t))
        cfg   = _make_cfg()
        result = find_min_cooling_years(
            R=0.3,
            loading_fraction=0.05,
            properties=props,
            rho_base=_RHO,
            repo_type="Bentonite",
            cfg=cfg,
        )
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_cooling_never_safe_returns_inf(self):
        """Near-constant source: Q_vol > Q_allowable even at t = 1000 yr → returns inf."""
        # λ = 1e-5 yr⁻¹ (half-life ≈ 69 000 yr); large amplitude and loading
        # Q_vol(1000) ≈ 100 × exp(-0.01) × 3571 × 0.3 ≈ 106 000 W/m³ >> 2000 W/m³
        props = _make_props(decay_func=lambda t: 100.0 * np.exp(-1e-5 * t))
        cfg   = _make_cfg()
        result = find_min_cooling_years(
            R=0.3,
            loading_fraction=0.3,
            properties=props,
            rho_base=_RHO,
            repo_type="Bentonite",
            cfg=cfg,
        )
        assert np.isinf(result) and result > 0, f"Expected +inf, got {result}"

    def test_cooling_years_closed_form(self):
        """
        Single-term decay → closed-form crossing time.

        Parameters:
          A = 50 W/kg, λ = 0.5 yr⁻¹, loading = 0.1, ρ_base = 2500 kg/m³
          → ρ_eff = 2777.8 kg/m³,  Q_vol(t) = 13888.9 × exp(−0.5t)  [W/m³]
          R = 0.3 m, Bentonite (surface limit 100°C), h_passive = 5, k = 2.0

        Q_allowable (surface-limited):
          Q_surf = (373.15 − 313.15) / (0.3 / 10) = 2000 W/m³

        Closed form:
          t_min = (1/λ) × ln(A·ρ_eff·loading / Q_allow)
                = 2 × ln(13888.9 / 2000)
                ≈ 3.876 yr

        Tolerance: ±0.1 yr.
        """
        props = _make_props(decay_func=lambda t: 50.0 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        result = find_min_cooling_years(
            R=0.3,
            loading_fraction=0.1,
            properties=props,
            rho_base=_RHO,
            repo_type="Bentonite",
            cfg=cfg,
        )

        # Analytical
        lam      = 0.5
        A        = 50.0
        eff_rho  = _RHO / (1.0 - 0.1)   # 2777.8
        loading  = 0.1
        # Q_allowable: surface-limited at R=0.3, h=5, T_surf_limit=100°C
        # k at midpoint ≈ 2.0 (constant); surface constraint is binding
        T_inf_K  = 40.0 + KELVIN_OFFSET
        T_surf_K = 100.0 + KELVIN_OFFSET
        Q_allow  = (T_surf_K - T_inf_K) / (0.3 / (2.0 * 5.0))   # = 2000.0
        t_analytical = (1.0 / lam) * math.log(A * eff_rho * loading / Q_allow)

        assert np.isfinite(result), f"Expected finite t_min, got {result}"
        assert abs(result - t_analytical) <= 0.1, (
            f"t_min = {result:.4f} yr, analytical = {t_analytical:.4f} yr "
            f"(diff = {abs(result - t_analytical):.4f} yr)"
        )

    def test_cooling_years_increases_with_loading(self):
        """Higher waste loading → longer required cooling time."""
        props = _make_props(decay_func=lambda t: 50.0 * np.exp(-0.5 * t))
        cfg   = _make_cfg()
        t_results = []
        for loading in [0.05, 0.10]:
            t = find_min_cooling_years(
                R=0.3,
                loading_fraction=loading,
                properties=props,
                rho_base=_RHO,
                repo_type="Bentonite",
                cfg=cfg,
            )
            t_results.append(t)

        assert all(np.isfinite(t) for t in t_results), (
            f"Both cooling times should be finite: {t_results}"
        )
        assert t_results[0] < t_results[1], (
            f"Cooling time should increase with loading: {t_results}"
        )
