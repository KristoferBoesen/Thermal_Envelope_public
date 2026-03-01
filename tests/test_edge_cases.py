"""Edge-case and user-error scenarios for the pipeline.

Maps common user mistakes to expected (non-crashing) behaviour:

    User action                     Failure mode                 Expected result
    ──────────────────────────────────────────────────────────────────────────────
    loading = 0 %                   Q_vol = 0 everywhere         h_min=nan, t_min=0.0
    loading = 99.9 %                ρ_eff → very large           returns inf/nan, no crash
    surface_limit < ambient_temp    Q_allowable < 0              t_min = inf
    R = 0.005 m, 5 nodes            coarse grid                  no exception
    Single-term decay               1 parameter fewer            t_min finite & converges
    Half-life ≈ 1e5 yr              Q always > Q_allow in window t_min = inf, returned fast
"""

import numpy as np
import pytest

from thermal_envelope.analysis.pipeline import find_min_h_active, find_min_cooling_years

# ---------------------------------------------------------------------------
# Shared helpers (identical to test_pipeline.py)
# ---------------------------------------------------------------------------
_K   = 2.0
_CP  = 500.0
_RHO = 2500.0


def _make_props(decay_func=None):
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


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_loading_passive_sufficient(self):
        """
        Loading = 0 → Q_vol = 0 everywhere.

        find_min_h_active: T_center = T_inf < T_limit → passive sufficient → nan.
        find_min_cooling_years: g(0) = −Q_allowable < 0 → immediately safe → 0.0.
        """
        props = _make_props()
        cfg   = _make_cfg()

        h_result = find_min_h_active(
            R=0.2,
            loading_fraction=0.0,
            properties=props,
            rho_base=_RHO,
            cfg=cfg,
        )
        assert np.isnan(h_result), (
            f"loading=0 should give nan for h_min, got {h_result}"
        )

        t_result = find_min_cooling_years(
            R=0.2,
            loading_fraction=0.0,
            properties=props,
            rho_base=_RHO,
            repo_type="Bentonite",
            cfg=cfg,
        )
        assert t_result == 0.0, (
            f"loading=0 should give 0.0 for t_min, got {t_result}"
        )

    def test_unit_loading_does_not_crash(self):
        """
        loading = 0.999 → ρ_eff ≈ 2.5 × 10⁶ kg/m³ — enormous Q_vol.

        Must not raise any exception.  The result will be inf (infeasible) or
        nan (passive sufficient at some weird edge), but never a Python crash.
        """
        props = _make_props()
        cfg   = _make_cfg()

        try:
            h_result = find_min_h_active(
                R=0.2,
                loading_fraction=0.999,
                properties=props,
                rho_base=_RHO,
                cfg=cfg,
            )
        except Exception as exc:
            pytest.fail(
                f"find_min_h_active raised {type(exc).__name__} for loading=0.999: {exc}"
            )

        try:
            t_result = find_min_cooling_years(
                R=0.2,
                loading_fraction=0.999,
                properties=props,
                rho_base=_RHO,
                repo_type="Bentonite",
                cfg=cfg,
            )
        except Exception as exc:
            pytest.fail(
                f"find_min_cooling_years raised {type(exc).__name__} for loading=0.999: {exc}"
            )

        # At least one result should be inf (can't cool such an extreme case)
        assert np.isinf(h_result) or np.isinf(t_result) or np.isnan(h_result), (
            f"Expected inf or nan for loading=0.999; got h={h_result}, t={t_result}"
        )

    def test_surface_limit_below_ambient(self):
        """
        Surface limit < ambient temperature → Q_allowable < 0.

        g(t) = Q_vol(t) − Q_allowable ≥ 0 − negative > 0 for all t
        → find_min_cooling_years returns inf.
        """
        props = _make_props()
        cfg   = _make_cfg(surface_C=30.0, ambient_C=40.0)  # surface 30°C < ambient 40°C

        result = find_min_cooling_years(
            R=0.2,
            loading_fraction=0.1,
            properties=props,
            rho_base=_RHO,
            repo_type="Bentonite",
            cfg=cfg,
        )
        assert np.isinf(result) and result > 0, (
            f"Expected +inf when surface limit < ambient; got {result}"
        )

    def test_very_small_radius_does_not_crash(self):
        """
        R = 0.005 m with n_nodes = 5 (very coarse grid).

        The solver must not crash.  Result may be nan, inf, or a finite float.
        """
        props = _make_props()
        cfg   = _make_cfg(nodes=5)

        try:
            h_result = find_min_h_active(
                R=0.005,
                loading_fraction=0.1,
                properties=props,
                rho_base=_RHO,
                cfg=cfg,
            )
        except Exception as exc:
            pytest.fail(
                f"find_min_h_active crashed at R=0.005 m: {type(exc).__name__}: {exc}"
            )

        assert h_result is not None  # sanity — will always pass if no exception

    def test_single_decay_term_config(self):
        """
        Single-term decay λ(t) = 50·exp(−0.3t).

        The pipeline must converge and return a finite cooling time.
        """
        props = _make_props(decay_func=lambda t: 50.0 * np.exp(-0.3 * t))
        cfg   = _make_cfg()

        result = find_min_cooling_years(
            R=0.3,
            loading_fraction=0.1,
            properties=props,
            rho_base=_RHO,
            repo_type="Bentonite",
            cfg=cfg,
        )
        assert np.isfinite(result), (
            f"Expected finite cooling time for single-term decay, got {result}"
        )
        assert result >= 0.0, f"Cooling time should be non-negative, got {result}"

    def test_cooling_never_within_window(self):
        """
        λ ≈ 1/100 000 yr⁻¹ (half-life ≈ 69 000 yr) → Q barely decreases.

        find_min_cooling_years should return inf quickly
        (g(T_SEARCH_MAX=1000 yr) >> 0) without timing out.
        """
        import time

        props = _make_props(decay_func=lambda t: 200.0 * np.exp(-1e-5 * t))
        cfg   = _make_cfg()

        t_start = time.monotonic()
        result  = find_min_cooling_years(
            R=0.3,
            loading_fraction=0.3,
            properties=props,
            rho_base=_RHO,
            repo_type="Bentonite",
            cfg=cfg,
        )
        elapsed = time.monotonic() - t_start

        assert np.isinf(result) and result > 0, (
            f"Expected +inf for near-constant source, got {result}"
        )
        assert elapsed < 10.0, (
            f"inf detection took {elapsed:.2f} s (expected < 10 s)"
        )
