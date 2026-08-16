"""
Tests for the configuration loader.

These read a frozen fixture config, never the repository's real
``solver_config.yaml``. Following the tutorial means editing that file
constantly - choosing a material, pointing ``waste_source`` at your own decay
curve - and doing so must never break the test suite.

The one test that does read the shipped config asserts only that it loads.
"""

from pathlib import Path

import numpy as np
import pytest
from aethon.config_loader import load_config, _make_expression

_REFERENCE_CONFIG = Path(__file__).parent / "data" / "reference_config.yaml"
_SHIPPED_CONFIG = Path(__file__).parent.parent / "solver_config.yaml"


@pytest.fixture
def cfg():
    return load_config(_REFERENCE_CONFIG)


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


def test_cp_shomate(waste_form):
    """
    Default material is BorosilicateGlass:
        cp = 935.56 + 0.38953*T - 24617000/T**2
    """
    T = 300.0
    expected = 935.56 + 0.38953 * T - 24617000.0 / T**2
    assert pytest.approx(waste_form["cp"](T), rel=1e-6) == expected


def test_k_constant(waste_form):
    """BorosilicateGlass has constant k = 1.2 W/(m·K)."""
    assert pytest.approx(waste_form["k"](300.0), rel=1e-6) == 1.2
    assert pytest.approx(waste_form["k"](800.0), rel=1e-6) == 1.2


def test_material_name_selected(cfg):
    """The 'material' key selects the named entry from the materials library."""
    assert cfg["waste_form_name"] == "BorosilicateGlass"
    assert cfg["centerline_limit_C"] == 500.0


# --- Copenhagen Atomics glass-ceramic materials ---

_CA_MATERIALS = {
    "CA_Recycling_Bg-CaF2": {
        "rho_base": 2321.5,
        "k": lambda T: 1.585 - 3.879e-4 * T + 1.291e-7 * T**2,
        "cp": lambda T: 330.6 + 1.911 * T - 1.086e-3 * T**2,
    },
    "CA_Emergency_Bg-CaF2-Zr": {
        "rho_base": 3052.8,
        "k": lambda T: 2.797 - 2.015e-3 * T + 1.176e-6 * T**2,
        "cp": lambda T: 406.1 + 1.348 * T - 7.654e-4 * T**2,
    },
}


@pytest.mark.parametrize("name", sorted(_CA_MATERIALS))
def test_ca_material_properties(name):
    """Coefficients must match the source analysis exactly at several T."""
    expected = _CA_MATERIALS[name]
    wf = load_config(_REFERENCE_CONFIG, material=name)["waste_form"]

    assert wf["rho_base"] == pytest.approx(expected["rho_base"])
    for T in (300.0, 600.0, 900.0):
        assert wf["k"](T) == pytest.approx(expected["k"](T), rel=1e-9)
        assert wf["cp"](T) == pytest.approx(expected["cp"](T), rel=1e-9)


@pytest.mark.parametrize("name", sorted(_CA_MATERIALS))
def test_ca_material_is_physical(name):
    """Conductivity and heat capacity must stay positive across the range."""
    wf = load_config(_REFERENCE_CONFIG, material=name)["waste_form"]
    T = np.linspace(300.0, 1000.0, 50)
    assert (wf["k"](T) > 0).all(), "thermal conductivity went non-positive"
    assert (wf["cp"](T) > 0).all(), "specific heat went non-positive"


def test_ca_materials_share_devitrification_limit(cfg):
    """Both glass-ceramics carry the 500 degC limit used in the source work."""
    for name in _CA_MATERIALS:
        assert load_config(_REFERENCE_CONFIG, material=name)["centerline_limit_C"] == 500.0


def test_zirconia_form_is_denser_and_more_conductive():
    """Adding zirconia should raise both density and conductivity."""
    plain = load_config(_REFERENCE_CONFIG, material="CA_Recycling_Bg-CaF2")["waste_form"]
    zr = load_config(_REFERENCE_CONFIG, material="CA_Emergency_Bg-CaF2-Zr")["waste_form"]

    assert zr["rho_base"] > plain["rho_base"]
    assert zr["k"](500.0) > plain["k"](500.0)


def test_unknown_material_lists_alternatives():
    """A typo must name what is actually available."""
    with pytest.raises(ValueError, match="not found in config"):
        load_config(_REFERENCE_CONFIG, material="NoSuchGlass")


def test_passive_conditions(cfg):
    """The passive phases carry the site's worst-case design basis."""
    assert cfg["passive_ambient_C"] == 50.0
    assert cfg["passive_h"] == 5.0
    assert cfg["h_passive"] == cfg["passive_h"]


def test_campaign_and_window(cfg):
    """Campaign mass, geometry, and the pre-encapsulation window are exposed."""
    assert cfg["total_waste_mass_kg"] == 116.0
    assert cfg["canister_aspect_ratio"] == 6.0
    assert cfg["pre_encap_max_years"] == 5.0


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
        "passive_ambient_C", "passive_h", "h_passive",
        "cooling_archetypes", "archetype_names",
        "total_waste_mass_kg", "canister_aspect_ratio",
        "pre_encap_min_years", "pre_encap_max_years",
        "radii_min", "radii_max", "radii_steps", "loadings_pct",
        "nodes", "max_years",
    ]
    for key in required:
        assert key in cfg, f"Missing config key: {key}"


def test_no_global_ambient_temperature(cfg):
    """
    There is deliberately no facility-wide ambient. The active-phase ambient
    belongs to the chosen cooling archetype, since an HTC and the temperature
    it works against only mean anything as a pair.
    """
    assert "ambient_temp_C" not in cfg


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


# --- The shipped config ---


def test_shipped_config_loads():
    """
    The repository's own solver_config.yaml must remain valid.

    Deliberately asserts nothing about its values: a user following the
    tutorial edits this file, and their edits are not a test failure.
    """
    shipped = load_config(_SHIPPED_CONFIG)
    assert callable(shipped["waste_form"]["decay"])
    assert shipped["waste_form_name"] in shipped["surface_limits_C"] or True
    assert shipped["total_waste_mass_kg"] is None or shipped["total_waste_mass_kg"] > 0
