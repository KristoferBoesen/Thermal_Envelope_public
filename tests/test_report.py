"""
Tests for the design-space maps.

The figures are the primary output, so the pieces that decide what a reader
actually sees are pinned here: which contour levels get chosen, and whether the
long-format results reshape onto the grid correctly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aethon.design.report import (
    choose_levels,
    format_years,
    pivot_field,
    plot_design_maps,
    sweep_stats,
)

_LADDER = {
    0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0,
    50.0, 100.0, 250.0, 500.0, 1000.0,
}


def _make_results(radii=(0.1, 0.2, 0.3), loadings=(5.0, 10.0, 15.0)):
    """A small complete sweep result, two geologies and two technologies."""
    rows = []
    for geology, offset in (("Bentonite", 4.0), ("Salt", 1.0)):
        for arch, scale in (("NaturalAir", 2.0), ("WaterPool", 1.0)):
            for R in radii:
                for loading in loadings:
                    heat = R * loading
                    rows.append({
                        "Geology": geology,
                        "Archetype": arch,
                        "Material": "TestGlass",
                        "Radius_m": R,
                        "Loading_Pct": loading,
                        "N_canisters": max(1, int(100 / heat)),
                        "t_coolers_off_yr": heat,
                        "t_geo_yr": heat * offset,
                        "Binding_At_Geo": "surface",
                        "t_encap_yr": heat * scale * 0.1,
                        "t_active_yr": max(0.0, heat - heat * scale * 0.1),
                        "Feasible": True,
                        "h_active": 5.0 if arch == "NaturalAir" else 750.0,
                        "T_ambient_active_C": 40.0,
                        "h_passive": 5.0,
                        "T_ambient_passive_C": 50.0,
                        "Q_per_canister_W": 100.0,
                        "Facility_Duty_W": 1000.0,
                    })
    return pd.DataFrame(rows)


class TestChooseLevels:

    def test_levels_are_round_numbers(self):
        """The whole point: a reader traces a '5 yr' line, not a '4.68 yr' one."""
        values = np.geomspace(0.09, 177.0, 200)
        assert set(choose_levels(values)).issubset(_LADDER)

    def test_levels_lie_inside_the_data(self):
        values = np.geomspace(0.3, 40.0, 100)
        for level in choose_levels(values):
            assert 0.3 < level < 40.0

    def test_levels_are_ascending_and_distinct(self):
        levels = choose_levels(np.geomspace(0.05, 500.0, 300))
        assert levels == sorted(levels)
        assert len(levels) == len(set(levels))

    def test_wide_range_is_spread_not_clustered(self):
        """
        Log spacing exists so a three-decade range does not put every line in
        one corner. The largest level must be far above the smallest.
        """
        levels = choose_levels(np.geomspace(0.1, 1000.0, 400))
        assert len(levels) >= 3
        assert levels[-1] / levels[0] > 20.0

    def test_narrow_range_still_returns_something(self):
        """A range containing no ladder rung must not silently draw nothing."""
        levels = choose_levels(np.linspace(3.1, 3.9, 50))
        assert len(levels) >= 2
        for level in levels:
            assert 3.1 <= level <= 3.9

    def test_constant_field_yields_no_levels(self):
        assert choose_levels(np.full(20, 5.0)) == []

    def test_empty_and_non_finite_input_is_safe(self):
        assert choose_levels(np.array([])) == []
        assert choose_levels(np.array([np.nan, np.inf])) == []


class TestFormatYears:

    @pytest.mark.parametrize("value,expected", [
        (0.25, "0.25 yr"), (1.0, "1 yr"), (2.5, "2.5 yr"),
        (10.0, "10 yr"), (100.0, "100 yr"),
    ])
    def test_labels_have_no_trailing_zeros(self, value, expected):
        assert format_years(value) == expected


class TestPivotField:

    def test_shape_matches_the_grid(self):
        df = _make_results()
        subset = df[(df["Geology"] == "Salt") & (df["Archetype"] == "WaterPool")]
        radii, loadings, Z = pivot_field(subset, "t_geo_yr")

        assert radii.shape == (3,)
        assert loadings.shape == (3,)
        assert Z.shape == (len(loadings), len(radii))

    def test_axes_come_back_sorted(self):
        df = _make_results(radii=(0.3, 0.1, 0.2))
        subset = df[df["Archetype"] == "WaterPool"]
        radii, loadings, _ = pivot_field(subset, "t_geo_yr")

        assert list(radii) == sorted(radii)
        assert list(loadings) == sorted(loadings)

    def test_infinities_become_nan(self):
        """contour cannot handle inf; an unreachable milestone must go blank."""
        df = _make_results()
        df.loc[0, "t_geo_yr"] = np.inf
        _, _, Z = pivot_field(df[df["Archetype"] == "NaturalAir"], "t_geo_yr")

        assert not np.isinf(Z).any()
        assert np.isnan(Z).any()

    def test_result_is_writable(self):
        """A read-only view from pandas would break the inf-to-nan cleanup."""
        _, _, Z = pivot_field(_make_results(), "t_geo_yr")
        Z[0, 0] = 1.0  # must not raise


class TestPlotDesignMaps:

    def test_writes_both_maps(self, tmp_path):
        paths = plot_design_maps(_make_results(), tmp_path, "TestGlass")
        assert len(paths) == 2
        for path in paths:
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_empty_results_write_nothing(self, tmp_path):
        assert plot_design_maps(pd.DataFrame(), tmp_path, "TestGlass") == []

    def test_single_loading_does_not_crash(self, tmp_path):
        """
        A quick run with one loading cannot be contoured, but must degrade to
        a figure-free result rather than raising.
        """
        df = _make_results(loadings=(10.0,))
        plot_design_maps(df, tmp_path, "TestGlass")  # must not raise

    def test_infeasible_designs_do_not_break_the_map(self, tmp_path):
        df = _make_results()
        df.loc[df["Archetype"] == "NaturalAir", "Feasible"] = False
        df.loc[df["Archetype"] == "NaturalAir", "t_encap_yr"] = np.inf
        assert len(plot_design_maps(df, tmp_path, "TestGlass")) == 2


class TestSweepStats:

    def test_reports_grid_extent(self):
        stats = sweep_stats(_make_results())
        assert stats["n_designs"] == 9
        assert stats["radius_min"] == pytest.approx(0.1)
        assert stats["radius_max"] == pytest.approx(0.3)
        assert stats["loading_min"] == pytest.approx(5.0)
        assert stats["loading_max"] == pytest.approx(15.0)

    def test_counts_feasibility_per_technology(self):
        df = _make_results()
        df.loc[df["Archetype"] == "NaturalAir", "Feasible"] = False
        stats = sweep_stats(df)

        by_name = {e["name"]: e for e in stats["archetypes"]}
        assert by_name["NaturalAir"]["n_feasible"] == 0
        assert by_name["NaturalAir"]["earliest_encap"] is None
        assert by_name["WaterPool"]["n_feasible"] == 9
        assert by_name["WaterPool"]["share_pct"] == pytest.approx(100.0)

    def test_reports_t_geo_range_per_geology(self):
        stats = sweep_stats(_make_results())
        by_name = {e["name"]: e for e in stats["geologies"]}
        # Bentonite carries a 4x offset in the fixture, so it must be later
        assert by_name["Bentonite"]["t_geo_max"] > by_name["Salt"]["t_geo_max"]

    def test_unreachable_geology_reports_none_not_nan(self):
        df = _make_results()
        df.loc[df["Geology"] == "Salt", "t_geo_yr"] = np.inf
        stats = sweep_stats(df)

        by_name = {e["name"]: e for e in stats["geologies"]}
        assert by_name["Salt"]["t_geo_min"] is None

    def test_empty_results_give_empty_stats(self):
        assert sweep_stats(pd.DataFrame()) == {}

    def test_counts_unexpected_binding_constraint(self):
        df = _make_results()
        df["Binding_At_Geo"] = "centre"
        assert sweep_stats(df)["n_unexpected_binding"] > 0

    def test_no_unexpected_binding_in_the_normal_case(self):
        assert sweep_stats(_make_results())["n_unexpected_binding"] == 0
