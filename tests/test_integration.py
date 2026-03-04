"""End-to-end smoke tests and a real-inventory integration test.

Fast tests (~1–5 s each) verify the CLI produces valid CSV output.
The slow test (@pytest.mark.slow, ~30–60 s) exercises the full preprocessor
pipeline with the real Optimistic_composition_5y.csv inventory.

Running only the fast tests::

    pytest tests/ -v -m "not slow"

Running all tests::

    pytest tests/ -v

Note: pytest will emit a warning about the unknown 'slow' marker unless
markers are registered.  To suppress it, add to a conftest.py or pytest.ini::

    [pytest]
    markers =
        slow: marks slow-running integration tests
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_envelope.config_loader import load_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TEST_DIR      = Path(__file__).parent
_REPO_ROOT     = _TEST_DIR.parent
_INVENTORY_CSV = _TEST_DIR / "data" / "Optimistic_composition_5y.csv"
_CHAIN_XML     = _REPO_ROOT / "chain_endfb71_pwr.xml"


# ---------------------------------------------------------------------------
# Fast smoke tests
# ---------------------------------------------------------------------------

class TestSmokeRun:

    @pytest.fixture(scope="class")
    def result_dir_and_df(self, tmp_path_factory):
        """
        Run main() once with a tiny grid (3 radii, 1 loading) and return the
        output directory and the resulting DataFrame.
        """
        from thermal_envelope.__main__ import main

        tmp_dir = tmp_path_factory.mktemp("smoke_results")
        main([
            "--no-plot",
            "--radii-steps", "3",
            "--loadings", "5",
            "--output-dir", str(tmp_dir),
        ])

        cfg       = load_config()
        label     = cfg["waste_form_name"]
        csv_name  = f"Design_Envelope_{label}_Bentonite.csv"
        csv_path  = tmp_dir / csv_name
        df        = pd.read_csv(csv_path) if csv_path.exists() else None
        return tmp_dir, df, csv_path

    def test_smoke_run_no_plot(self, result_dir_and_df):
        """main() runs without exception and produces a CSV file."""
        _, _, csv_path = result_dir_and_df
        assert csv_path.exists(), f"Expected CSV at {csv_path}"

    def test_csv_has_expected_columns(self, result_dir_and_df):
        """Output CSV has required columns and at least one non-NaN entry."""
        _, df, _ = result_dir_and_df
        assert df is not None, "CSV was not produced"
        required = {"Radius_m", "Loading_Pct", "Min_H_Active", "Min_Cooling_Years"}
        assert required.issubset(df.columns), (
            f"Missing columns; got {list(df.columns)}"
        )
        # At least one column should have a real value somewhere
        assert not (df["Radius_m"].isna().all()), "Radius_m column is all NaN"

    def test_h_min_increases_with_radius(self, result_dir_and_df):
        """
        For a given loading, finite h_min values must be non-decreasing with R.

        Physics: larger radius → greater thermal resistance → more cooling needed.
        """
        _, df, _ = result_dir_and_df
        assert df is not None, "CSV was not produced"

        for loading, grp in df.groupby("Loading_Pct"):
            grp = grp.sort_values("Radius_m")
            finite_h = grp["Min_H_Active"].dropna().values
            if len(finite_h) < 2:
                continue   # need at least 2 finite values to check monotonicity
            diffs = np.diff(finite_h)
            assert (diffs >= -1e-6).all(), (
                f"Loading {loading}%: h_min decreased with R — diffs = {diffs}"
            )


# ---------------------------------------------------------------------------
# Slow integration test: real inventory + chain file
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRealInventory:

    def test_preprocessor_real_inventory(self):
        """
        Full preprocessor pipeline on the real Copenhagen Atomics inventory.

        Assertions:
          - More than 50 isotopes matched to the chain
          - Max specific power > 0  (regression guard for the Q-value bug)
          - Power is broadly decreasing after the first month
          - Fitted R² > 0.95
          - All fitted term amplitudes are finite and positive
        """
        if not _INVENTORY_CSV.exists():
            pytest.skip(f"Inventory file not found: {_INVENTORY_CSV}")
        if not _CHAIN_XML.exists():
            pytest.skip(f"Chain file not found: {_CHAIN_XML}")

        from decay_preprocessor.chain_parser import parse_chain
        from decay_preprocessor.bateman_solver import solve_decay
        from decay_preprocessor.decay_fitter import fit_decay_curve

        # Parse chain
        nuc_to_idx, decay_constants, q_values, matrix_A = parse_chain(_CHAIN_XML)

        # Load inventory
        inventory_df = pd.read_csv(_INVENTORY_CSV, comment="#")
        n_matched = sum(
            1 for iso in inventory_df["Isotope"] if iso in nuc_to_idx
        )
        assert n_matched > 50, (
            f"Only {n_matched} isotopes matched the chain — "
            "expected > 50 for a realistic inventory"
        )

        # Total mass from header comment or use a nominal value
        sample_mass_kg = 115.984   # from the inventory header

        # Solve for 10 years
        result_df = solve_decay(
            inventory_df=inventory_df,
            nuc_to_idx=nuc_to_idx,
            decay_constants=decay_constants,
            q_values=q_values,
            matrix_A=matrix_A,
            sample_mass_kg=sample_mass_kg,
            duration_years=10.0,
        )

        Q = result_df["Specific_Power_W_kg"].values
        t = result_df["Time_Years"].values

        # Guard against the original Q-value bug (all-zero power)
        assert Q.max() > 0.0, (
            "Max specific power is zero — possible Q-value regression."
        )

        # Power should generally decrease after 1 month
        post_month = Q[t > 1 / 12]
        assert len(post_month) > 10
        # Mean of last 10% of points should be less than mean of first 10%
        n10 = max(1, len(post_month) // 10)
        assert post_month[-n10:].mean() < post_month[:n10].mean(), (
            "Power does not decrease over time — unexpected physics."
        )

        # Fit and check quality
        terms, r2, _ = fit_decay_curve(t, Q)
        assert r2 > 0.95, f"Fit R² = {r2:.4f} (expected > 0.95)"
        for i, (A, lam) in enumerate(terms):
            assert np.isfinite(A) and A > 0, (
                f"Term {i}: amplitude A = {A} is not finite and positive"
            )
