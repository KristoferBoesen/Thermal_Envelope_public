"""Unit tests for the Bateman decay chain solver.

Verifies solve_decay against known analytical solutions for simple 1- and
2-nuclide chains built from synthetic XML strings (no dependency on the real
27 MB chain file).

Analytical solutions used
--------------------------
Single nuclide A → stable B (purely exponential):
    N_A(t) = N₀ · exp(−λ_A · t)
    P(t)   = λ_A · N_A(t) · E_A · eV_TO_J  [W]

Two-nuclide chain A → B → stable C (Bateman):
    N_B(t) = N_A0 · λ_A / (λ_B − λ_A) · (exp(−λ_A·t) − exp(−λ_B·t))

    (valid for λ_A ≠ λ_B, N_B(0) = 0)

To extract N_B from solve_decay output without direct access to the
atom-count matrix, we set q_A = 0 and q_B > 0 so that only B contributes
to the heat output:
    P(t) = λ_B · N_B(t) · q_B · eV_TO_J
    → N_B(t) = P(t) / (λ_B · q_B · eV_TO_J)
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import scipy.interpolate as interp

from decay_preprocessor.chain_parser import parse_chain
from decay_preprocessor.bateman_solver import solve_decay

_LN2       = 0.6931471805599453
_EV_TO_J   = 1.60218e-19
_SEC_PER_Y = 365.25 * 24 * 3600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_xml_string(xml_str):
    """Write xml_str to a temp file, parse it, return the result."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        f.write(xml_str)
        tmp_path = f.name
    try:
        result = parse_chain(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result


def _single_nuclide_xml(half_life_s, decay_energy_eV):
    """Build a minimal <depletion_chain> XML for one radioactive nuclide."""
    return f"""<depletion_chain>
  <nuclide name="A" half_life="{half_life_s}" decay_energy="{decay_energy_eV}">
    <decay type="beta-" target="B" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="B"/>
</depletion_chain>"""


def _two_nuclide_xml(half_life_A_s, half_life_B_s):
    """
    Two-nuclide chain A→B→stable C.

    q_A = 0, q_B = 1000 eV so only B contributes to power, allowing N_B
    to be back-calculated from the heat output.
    """
    return f"""<depletion_chain>
  <nuclide name="A" half_life="{half_life_A_s}" decay_energy="0.0">
    <decay type="beta-" target="B" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="B" half_life="{half_life_B_s}" decay_energy="1000.0">
    <decay type="beta-" target="C" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="C"/>
</depletion_chain>"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSolveDecay:

    # Half-life chosen so the nuclide decays appreciably within 0.25 yr
    _HALF_LIFE_S = 1e6   # ≈ 11.6 days
    _LAM         = _LN2 / _HALF_LIFE_S
    _Q_EV        = 1000.0
    _N0          = 1e25
    _MASS        = 1.0   # kg

    @pytest.fixture(scope="class")
    def single_nuc_result(self):
        """Solve decay for a single-nuclide A→stable B chain."""
        xml = _single_nuclide_xml(self._HALF_LIFE_S, self._Q_EV)
        nuc_to_idx, decay_constants, q_values, matrix_A = _parse_xml_string(xml)
        inventory = pd.DataFrame({"Isotope": ["A"], "Atoms": [self._N0]})
        df = solve_decay(
            inventory_df=inventory,
            nuc_to_idx=nuc_to_idx,
            decay_constants=decay_constants,
            q_values=q_values,
            matrix_A=matrix_A,
            sample_mass_kg=self._MASS,
            duration_years=0.25,   # three half-lives
            n_points=500,
        )
        return df, decay_constants[nuc_to_idx["A"]]

    def test_single_nuclide_exponential_decay(self, single_nuc_result):
        """
        Power output should follow P ∝ exp(−λ·t) within 1% at three time points.

        Uses a ratio check between consecutive interpolated points so no exact
        time alignment to the log-spaced grid is needed.
        """
        df, lam = single_nuc_result
        t_s    = df["Time_Years"].values * _SEC_PER_Y
        P      = df["Heat_Watts"].values

        # Check exponential ratios at ~25%, 50%, 75% of the simulation range
        n = len(t_s)
        for i in [n // 4, n // 2, 3 * n // 4]:
            j = i + 1
            dt = t_s[j] - t_s[i]
            expected_ratio = np.exp(-lam * dt)
            actual_ratio   = P[j] / P[i]
            assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, (
                f"Ratio mismatch at index {i}: "
                f"expected {expected_ratio:.6f}, got {actual_ratio:.6f}"
            )

    def test_specific_power_matches_analytic(self, single_nuc_result):
        """
        Absolute value: P(t₀) = λ·N₀·q·eV_TO_J within 1%.

        The very first evaluation point is close to t=0, so N_A ≈ N₀.
        """
        df, lam = single_nuc_result
        t0_s         = df["Time_Years"].values[0] * _SEC_PER_Y
        P_analytic   = lam * self._N0 * np.exp(-lam * t0_s) * self._Q_EV * _EV_TO_J
        P_solver     = df["Heat_Watts"].values[0]
        assert abs(P_solver - P_analytic) / P_analytic < 0.01, (
            f"P_solver = {P_solver:.4e} W, P_analytic = {P_analytic:.4e} W"
        )

    def test_two_nuclide_chain_daughter_production(self):
        """
        N_B(t) from the solver matches the Bateman two-nuclide formula within 1%.

        Chain: A(t½=1e5 s) → B(t½=1e6 s) → stable C.
        N_B is back-calculated from heat output since q_A = 0, q_B = 1000 eV.
        """
        half_life_A_s = 1e5   # A decays fast (t½ ≈ 1.16 days)
        half_life_B_s = 1e6   # B decays slowly (t½ ≈ 11.6 days)
        lam_A = _LN2 / half_life_A_s
        lam_B = _LN2 / half_life_B_s
        N0_A  = 1e25
        q_B   = 1000.0   # eV

        xml = _two_nuclide_xml(half_life_A_s, half_life_B_s)
        nuc_to_idx, decay_constants, q_values, matrix_A = _parse_xml_string(xml)

        inventory = pd.DataFrame({"Isotope": ["A"], "Atoms": [N0_A]})
        df = solve_decay(
            inventory_df=inventory,
            nuc_to_idx=nuc_to_idx,
            decay_constants=decay_constants,
            q_values=q_values,
            matrix_A=matrix_A,
            sample_mass_kg=1.0,
            duration_years=0.05,   # covers several half-lives of A
            n_points=500,
        )

        t_s = df["Time_Years"].values * _SEC_PER_Y
        P   = df["Heat_Watts"].values

        # Back-calculate N_B from power (only B contributes since q_A = 0)
        N_B_solver = P / (lam_B * q_B * _EV_TO_J)

        # Bateman formula
        N_B_analytical = (
            N0_A * (lam_A / (lam_B - lam_A))
            * (np.exp(-lam_A * t_s) - np.exp(-lam_B * t_s))
        )

        # Avoid t ≈ 0 where N_B ≈ 0 (relative error undefined); check mid-range
        mask = t_s > half_life_A_s
        assert mask.sum() > 10, "Need enough points after one half-life of A"

        rel_err = np.abs(N_B_solver[mask] - N_B_analytical[mask]) / N_B_analytical[mask]
        assert rel_err.max() < 0.01, (
            f"Max relative error in N_B = {rel_err.max():.4f} (>1%)"
        )

    def test_zero_inventory_zero_power(self):
        """All atom counts = 0 → specific power must be exactly zero everywhere."""
        xml = _single_nuclide_xml(1e6, 1000.0)
        nuc_to_idx, decay_constants, q_values, matrix_A = _parse_xml_string(xml)
        inventory = pd.DataFrame({"Isotope": ["A"], "Atoms": [0.0]})
        df = solve_decay(
            inventory_df=inventory,
            nuc_to_idx=nuc_to_idx,
            decay_constants=decay_constants,
            q_values=q_values,
            matrix_A=matrix_A,
            sample_mass_kg=1.0,
            duration_years=1.0,
        )
        assert (df["Specific_Power_W_kg"].values == 0.0).all(), (
            "Non-zero power from empty inventory"
        )

    def test_output_dataframe_structure(self, single_nuc_result):
        """Output DataFrame must have required columns, monotone time, non-negative power."""
        df, _ = single_nuc_result
        assert set(df.columns) >= {"Time_Years", "Heat_Watts", "Specific_Power_W_kg"}, (
            f"Missing columns; got {list(df.columns)}"
        )
        t = df["Time_Years"].values
        assert (np.diff(t) > 0).all(), "Time_Years is not strictly increasing"
        assert (df["Heat_Watts"].values >= 0.0).all(), "Negative heat detected"
        assert (df["Specific_Power_W_kg"].values >= 0.0).all(), "Negative specific power"

    def test_dimensional_consistency(self, single_nuc_result):
        """Heat_Watts / sample_mass_kg == Specific_Power_W_kg at every point."""
        df, _ = single_nuc_result
        ratio = df["Heat_Watts"].values / self._MASS
        np.testing.assert_allclose(
            ratio,
            df["Specific_Power_W_kg"].values,
            rtol=1e-10,
            err_msg="Heat_Watts / mass ≠ Specific_Power_W_kg",
        )
