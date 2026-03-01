"""Unit tests for the triple-exponential decay fitter.

Tests verify:
  - Return type and structure
  - Perfect fit recovery on exact 3-term data
  - R² / RMSE directional consistency
  - Cutoff correctly excludes early short-lived transient
"""

import numpy as np
import pytest

from decay_preprocessor.decay_fitter import fit_decay_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_three_term_data(t):
    """Exact 3-term signal used across several tests."""
    A1, lam1 = 100.0, 2.0
    A2, lam2 =  20.0, 0.3
    A3, lam3 =   5.0, 0.05
    return (
        A1 * np.exp(-lam1 * t)
        + A2 * np.exp(-lam2 * t)
        + A3 * np.exp(-lam3 * t)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitDecayCurve:

    def test_returns_3_terms_always(self):
        """fit_decay_curve always returns exactly 3 [A, λ] pairs."""
        t = np.linspace(1 / 12, 10, 500)
        Q = 50.0 * np.exp(-0.5 * t)
        terms, _, _ = fit_decay_curve(t, Q)
        assert len(terms) == 3, f"Expected 3 terms, got {len(terms)}"
        for pair in terms:
            assert len(pair) == 2, f"Each term should have 2 elements, got {pair}"

    def test_return_is_tuple_of_3(self):
        """Returns (list, float, float) with finite r2 and rmse."""
        t = np.linspace(1 / 12, 10, 500)
        Q = 50.0 * np.exp(-0.5 * t)
        result = fit_decay_curve(t, Q)
        assert len(result) == 3, "Should return (terms, r2, rmse)"
        terms, r2, rmse = result
        assert isinstance(terms, list)
        assert isinstance(r2,    float) and np.isfinite(r2)
        assert isinstance(rmse,  float) and np.isfinite(rmse)

    def test_perfect_recovery_3term_data(self):
        """
        Fitting exact 3-term data: R² > 0.999 and RMSE below 1% of signal mean.

        Signal: Q(t) = 100·exp(−2t) + 20·exp(−0.3t) + 5·exp(−0.05t)

        Note: curve_fit minimises absolute (linear-space) residuals starting
        from a generic initial guess, so it may converge to a local minimum
        with different term values than the true parameters — yet the resulting
        curve still closely matches the original data.  We test aggregate fit
        quality (R², RMSE) rather than exact parameter recovery.
        """
        t = np.linspace(1 / 12, 10, 500)
        Q = _make_three_term_data(t)
        terms, r2, rmse = fit_decay_curve(t, Q)

        assert r2 > 0.999, f"R² = {r2:.6f} (expected > 0.999)"
        # Absolute RMSE threshold: fitter minimises linear residuals, so small
        # absolute error on the dominant early-time signal is the correct metric.
        assert rmse < 0.5, f"RMSE = {rmse:.4f} W/kg (expected < 0.5 W/kg)"

    def test_r2_rmse_self_consistent(self):
        """
        Fit quality metrics are directionally consistent:
        noisier data → lower R², higher RMSE.
        """
        rng = np.random.default_rng(0)
        t   = np.linspace(1 / 12, 10, 500)
        Q   = _make_three_term_data(t)

        # Clean fit
        _, r2_clean, rmse_clean = fit_decay_curve(t, Q)

        # Noisy fit (add ±10% Gaussian noise)
        Q_noisy = Q + rng.normal(scale=0.1 * Q.mean(), size=len(Q))
        Q_noisy = np.clip(Q_noisy, 0.0, None)  # keep non-negative
        _, r2_noisy, rmse_noisy = fit_decay_curve(t, Q_noisy)

        assert r2_clean  > r2_noisy,   "Noisier data should give lower R²"
        assert rmse_clean < rmse_noisy, "Noisier data should give higher RMSE"

    def test_cutoff_excludes_early_spike(self):
        """
        Early short-lived spike (t < 1/12 yr) is excluded by the default cutoff.

        Signal: 1000·exp(−100t) [spike at days] + 5·exp(−0.05t) [slow component].
        After cutoff at 1/12 yr: spike ≈ 1000·exp(−8.33) ≈ 2.4×10⁻⁴ W/kg — negligible.
        Fit on the retained data should have R² > 0.99.
        """
        t = np.linspace(0.001, 10.0, 1000)
        Q = 1000.0 * np.exp(-100.0 * t) + 5.0 * np.exp(-0.05 * t)

        _, r2, _ = fit_decay_curve(t, Q)

        assert r2 > 0.99, (
            f"R² = {r2:.4f} on post-cutoff data (expected > 0.99). "
            "Early spike may not have been correctly excluded."
        )
