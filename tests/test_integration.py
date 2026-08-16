"""End-to-end smoke tests and a real-inventory integration test.

Fast tests (~1–5 s each) verify the CLI produces valid CSV output.
The slow test (@pytest.mark.slow, ~30–60 s) exercises the full preprocessor
pipeline with the real MSR inventory shipped in examples/.

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

from aethon.config_loader import load_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TEST_DIR      = Path(__file__).parent
_REPO_ROOT     = _TEST_DIR.parent
_INVENTORY_CSV = _REPO_ROOT / "examples" / "msr_inventory_5y.csv"
_CHAIN_XML     = _REPO_ROOT / "chain_endfb71_pwr.xml"


# ---------------------------------------------------------------------------
# Fast smoke tests
# ---------------------------------------------------------------------------

class TestSmokeRun:

    @pytest.fixture(scope="class")
    def run_dir(self, tmp_path_factory):
        """
        Run the CLI on the smallest grid that still produces a contour.

        Every point costs a transient root-find of roughly a dozen FEM solves,
        and this test exists to check the wiring rather than the physics, so
        the grid is 3x3 - the least that a contour can be drawn through.
        """
        from aethon.__main__ import main

        tmp_dir = tmp_path_factory.mktemp("explore_results")
        exit_code = main([
            "--radii-min", "0.1",
            "--radii-max", "0.3",
            "--radii-steps", "3",
            "--loadings-min", "5",
            "--loadings-max", "15",
            "--loadings-steps", "3",
            "--repo", "Salt",
            "--archetype", "ForcedAir",
            "--output-dir", str(tmp_dir),
        ])
        assert exit_code == 0
        return tmp_dir

    @pytest.fixture(scope="class")
    def results(self, run_dir):
        label = load_config()["waste_form_name"]
        return pd.read_csv(run_dir / f"explore_full_{label}.csv")

    def test_results_csv_written(self, results):
        assert not results.empty

    def test_reports_all_milestones(self, results):
        """Every milestone and operating condition must reach the CSV."""
        required = {
            "Geology", "Archetype", "Material", "Radius_m", "Loading_Pct",
            "N_canisters", "t_encap_yr", "t_coolers_off_yr", "t_active_yr",
            "t_geo_yr", "Binding_At_Geo", "h_active", "T_ambient_active_C",
            "h_passive", "T_ambient_passive_C", "Q_per_canister_W",
            "Facility_Duty_W",
        }
        assert required.issubset(results.columns), (
            f"missing: {required - set(results.columns)}"
        )

    def test_sweep_is_complete(self, results):
        """3 radii x 3 loadings, no point dropped."""
        assert len(results) == 9
        assert results["t_geo_yr"].notna().all()

    def test_milestones_are_ordered(self, results):
        """t_encap <= t_coolers_off <= t_geo on every reported row."""
        assert (results["t_coolers_off_yr"] <= results["t_geo_yr"] + 1e-6).all()
        needs_cooling = results[results["t_active_yr"] > 0.0]
        assert (
            needs_cooling["t_encap_yr"]
            <= needs_cooling["t_coolers_off_yr"] + 1e-6
        ).all()

    def test_only_requested_options_appear(self, results):
        assert set(results["Geology"]) == {"Salt"}
        assert set(results["Archetype"]) == {"ForcedAir"}

    def test_both_maps_are_written(self, run_dir):
        label = load_config()["waste_form_name"]
        assert (run_dir / f"design_map_passive_{label}.png").exists()
        assert (run_dir / f"design_map_encapsulation_{label}.png").exists()

    def test_run_record_is_written(self, run_dir):
        assert (run_dir / "run_config.yaml").exists()

    def test_unknown_archetype_exits_nonzero(self, tmp_path):
        """A typo in a technology name must fail cleanly, not crash."""
        from aethon.__main__ import main

        assert main([
            "--no-plot", "--radii-steps", "2",
            "--archetype", "Cryogenic",
            "--output-dir", str(tmp_path),
        ]) == 1


class TestWasteSourceHandoff:

    def test_preprocessor_output_feeds_the_solver(self, tmp_path):
        """
        waste_source.yaml written by the preprocessor must be readable by
        load_config and supply the decay curve.
        """
        from decay_preprocessor.run_preprocessor import write_waste_source

        write_waste_source(
            path=tmp_path / "waste_source.yaml",
            terms=[(500.0, 4.0), (30.0, 0.3)],
            sample_mass_kg=100.0,
            r2=0.9995,
            rmse=1.2,
        )

        reference = _TEST_DIR / "data" / "reference_config.yaml"
        config_text = reference.read_text(encoding="utf-8")
        config_text += "\nwaste_source: waste_source.yaml\n"
        (tmp_path / "solver_config.yaml").write_text(config_text, encoding="utf-8")

        cfg = load_config(tmp_path / "solver_config.yaml")

        assert cfg["waste_form"]["decay"](0.0) == pytest.approx(530.0)

    def test_campaign_mass_comes_only_from_the_config(self, tmp_path):
        """
        The waste stream describes specific power, which is intensive and says
        nothing about how much waste exists. Campaign size therefore has one
        home - the config - and a waste_source file must not displace it.
        """
        from decay_preprocessor.run_preprocessor import write_waste_source

        write_waste_source(
            path=tmp_path / "waste_source.yaml",
            terms=[(500.0, 4.0)],
            sample_mass_kg=100.0,
            r2=0.999,
            rmse=1.0,
        )

        reference = _TEST_DIR / "data" / "reference_config.yaml"
        config_text = reference.read_text(encoding="utf-8")
        config_text += "\nwaste_source: waste_source.yaml\n"
        (tmp_path / "solver_config.yaml").write_text(config_text, encoding="utf-8")

        cfg = load_config(tmp_path / "solver_config.yaml")

        # 116.0 is the reference config's campaign value, not the 100.0 the
        # preprocessor normalised by
        assert cfg["total_waste_mass_kg"] == pytest.approx(116.0)


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
