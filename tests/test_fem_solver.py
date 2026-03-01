"""Tests for the transient FEM solver.

Uses simple inline material properties (constant k, cp; single-term decay)
so these tests are independent of solver_config.yaml content.
"""

import numpy as np
import pytest
from thermal_envelope.constants import KELVIN_OFFSET
from thermal_envelope.physics.fem_solver import WasteForm

# --- Inline properties for testing (no config dependency) ---

_RHO_BASE = 2500.0  # kg/m³

_PROPS = {
    "decay": lambda t: 50.0 * np.exp(-0.5 * t),   # W/kg, single-term
    "cp": lambda T: np.full_like(np.asarray(T, dtype=float), 500.0),   # J/(kg·K)
    "k": lambda T: np.full_like(np.asarray(T, dtype=float), 2.0),       # W/(m·K)
}

_T_INF_K = 40.0 + KELVIN_OFFSET   # 313.15 K


def _make_sim(R=0.2, h=10.0, loading=0.05, n_nodes=50):
    eff_rho = _RHO_BASE / (1.0 - loading)
    return WasteForm(
        R=R,
        ambient_T=_T_INF_K,
        h_coeff=h,
        loading_fraction=loading,
        properties=_PROPS,
        cooling_years=1.0 / 12.0,
        effective_density=eff_rho,
        n_nodes=n_nodes,
    )


def test_center_hotter_than_surface():
    """Peak centre temperature must exceed peak surface temperature."""
    sim = _make_sim(R=0.2, h=10.0, loading=0.10)
    _, T_center_K, T_surface_K = sim.solve_for_peak(max_years=50.0)
    assert T_center_K > T_surface_K


def test_higher_h_lowers_temperature():
    """Increasing h must decrease peak centre temperature."""
    temps = []
    for h in [5.0, 50.0, 500.0]:
        sim = _make_sim(R=0.2, h=h, loading=0.05)
        _, T_center_K, _ = sim.solve_for_peak(max_years=50.0)
        temps.append(T_center_K)
    assert temps[0] > temps[1] > temps[2], \
        f"Temperature should decrease with h: {temps}"


def test_uniform_initial_condition():
    """Initial condition: all nodes at T_inf."""
    sim = _make_sim(R=0.1, h=10.0, n_nodes=20)
    T0 = np.ones(sim.N) * sim.T_inf
    assert np.allclose(T0, _T_INF_K)


def test_higher_loading_raises_temperature():
    """Higher waste loading must produce a higher peak centre temperature."""
    temps = []
    for loading in [0.05, 0.10, 0.20]:
        sim = _make_sim(R=0.3, h=20.0, loading=loading)
        _, T_center_K, _ = sim.solve_for_peak(max_years=50.0)
        temps.append(T_center_K)
    assert temps[0] < temps[1] < temps[2], \
        f"Temperature should increase with loading: {temps}"
