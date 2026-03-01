"""Tests for the configuration loader."""

import numpy as np
import pytest
from thermal_envelope.config_loader import load_config, _make_expression


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def waste_form(cfg):
    return cfg["waste_form"]


def test_load_returns_waste_form(cfg):
    """Config must contain a single 'waste_form' dict."""
    assert "waste_form" in cfg


def test_waste_form_keys(waste_form):
    """waste_form must have rho_base, decay, cp, k."""
    assert "rho_base" in waste_form
    assert callable(waste_form["decay"]), "decay must be callable"
    assert callable(waste_form["cp"]), "cp must be callable"
    assert callable(waste_form["k"]), "k must be callable"


def test_waste_form_name(cfg):
    """waste_form_name must be a non-empty string."""
    assert "waste_form_name" in cfg
    assert isinstance(cfg["waste_form_name"], str)
    assert len(cfg["waste_form_name"]) > 0


def test_decay_at_zero(waste_form):
    """
    Default config decay_terms: [100.0, 5.0], [20.0, 0.5], [2.0, 0.05]
    decay(0) = 100 + 20 + 2 = 122.0 W/kg
    """
    result = waste_form["decay"](0.0)
    assert pytest.approx(result, rel=1e-6) == 122.0


def test_cp_polynomial(waste_form):
    """
    Default cp: "500.0 + 0.5*T"  =>  cp(300) = 500 + 0.5*300 = 650.0
    """
    expected = 500.0 + 0.5 * 300.0
    assert pytest.approx(waste_form["cp"](300.0), rel=1e-6) == expected


def test_k_polynomial(waste_form):
    """
    Default k: "2.0 - 1.0e-3*T"  =>  k(300) = 2.0 - 0.3 = 1.7
    """
    expected = 2.0 - 1e-3 * 300.0
    assert pytest.approx(waste_form["k"](300.0), rel=1e-6) == expected


def test_decay_monotonically_decreasing(waste_form):
    """Decay heat must decrease over time."""
    t = np.linspace(0, 100, 50)
    Q = [waste_form["decay"](ti) for ti in t]
    assert all(Q[i] >= Q[i + 1] for i in range(len(Q) - 1)), \
        "Decay heat is not monotonically decreasing"


def test_config_has_required_keys(cfg):
    """Config must contain all expected top-level keys."""
    required = [
        "waste_form_name", "waste_form",
        "centerline_limit_C", "safety_factor", "surface_limits_C",
        "ambient_temp_C", "h_passive", "radii_min", "radii_max",
        "radii_steps", "loadings_pct", "nodes", "max_years",
        "cooling_months",
    ]
    for key in required:
        assert key in cfg, f"Missing config key: {key}"


# --- _make_expression unit tests ---


def test_make_expression_constant():
    """Constant string returns a fixed value regardless of T."""
    f = _make_expression("500.0")
    assert pytest.approx(f(300.0), rel=1e-9) == 500.0
    assert pytest.approx(f(1000.0), rel=1e-9) == 500.0


def test_make_expression_constant_vectorised():
    """Constant expression returns 500.0 for all elements of a vectorised input."""
    f = _make_expression("500.0")
    T = np.array([300.0, 400.0, 500.0])
    result = f(T)
    np.testing.assert_allclose(np.broadcast_to(result, T.shape), 500.0)


def test_make_expression_linear():
    """Linear expression evaluates correctly at a scalar temperature."""
    f = _make_expression("500.0 + 0.5*T")
    assert pytest.approx(f(300.0), rel=1e-9) == 650.0


def test_make_expression_power_law():
    """Power-law expression matches direct numpy evaluation."""
    f = _make_expression("200.0 * T**0.35")
    T = 400.0
    assert pytest.approx(f(T), rel=1e-9) == 200.0 * T**0.35


def test_make_expression_numpy_function():
    """Expressions using np.* functions evaluate correctly."""
    f = _make_expression("np.sqrt(T)")
    assert pytest.approx(f(100.0), rel=1e-9) == 10.0


def test_make_expression_piecewise_inline():
    """np.interp piecewise expression interpolates correctly."""
    f = _make_expression("np.interp(T, [300, 500], [450, 550])")
    assert pytest.approx(f(400.0), rel=1e-9) == 500.0


def test_make_expression_vectorised_input():
    """Linear expression returns an ndarray when given an ndarray."""
    f = _make_expression("500.0 + 0.5*T")
    T = np.array([200.0, 300.0, 400.0])
    result = f(T)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [600.0, 650.0, 700.0])
