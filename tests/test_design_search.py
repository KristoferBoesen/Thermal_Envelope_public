"""
Tests for the three-milestone storage model and the design search.

The milestones are all measured from reactor shutdown:

    t_encap  <=  t_coolers_off  <=  t_geo

and the active cooling duration is ``t_coolers_off - t_encap``.  These tests
pin the ordering, the arithmetic tying the milestones together, and the
monotonic responses that make the results interpretable.
"""

import numpy as np
import pytest

from aethon.analysis.pipeline import find_min_encap_years, find_total_decay_years
from aethon.design.objectives import evaluate_cheap, evaluate_gate
from aethon.design.search import resolve_loadings, run_exploration

_RHO = 2500.0
_WEAK = {"h": 5.0, "ambient_C": 40.0}
_STRONG = {"h": 250.0, "ambient_C": 25.0}


def _make_props(decay_func=None):
    return {
        "rho_base": _RHO,
        "decay": decay_func or (lambda t: 200.0 * np.exp(-0.5 * t)),
        "cp": lambda T: np.full_like(np.asarray(T, dtype=float), 500.0),
        "k": lambda T: np.full_like(np.asarray(T, dtype=float), 2.0),
    }


def _make_cfg(**overrides):
    cfg = {
        "waste_form_name": "TestGlass",
        "waste_form": _make_props(),
        "centerline_limit_C": 400.0,
        "safety_factor": 1.0,
        "passive_ambient_C": 50.0,
        "passive_h": 5.0,
        "h_passive": 5.0,
        "surface_limits_C": {"Bentonite": 100.0, "Salt": 200.0},
        "cooling_archetypes": {},
        "total_waste_mass_kg": 500.0,
        "canister_aspect_ratio": 6.0,
        "pre_encap_min_years": 0.0,
        "pre_encap_max_years": 10.0,
        "nodes": 15,
        "max_years": 50.0,
        "radii_min": 0.05,
        "radii_max": 0.30,
        "radii_steps": 4,
        "loadings_pct": [5.0, 15.0],
    }
    cfg.update(overrides)
    return cfg


# ===========================================================================
# Milestone ordering and arithmetic
# ===========================================================================

class TestMilestoneOrdering:

    def test_centre_only_gate_is_never_later_than_combined(self):
        """
        Dropping the surface constraint can only relax the limit, so the
        coolers-off milestone cannot come after repository readiness.
        """
        props, cfg = _make_props(), _make_cfg()
        for R in (0.08, 0.15, 0.25):
            t_coolers_off = find_total_decay_years(
                R, 0.10, props, _RHO, cfg, surface_limit_C=np.inf,
            )
            t_geo = find_total_decay_years(
                R, 0.10, props, _RHO, cfg, surface_limit_C=100.0,
            )
            assert t_coolers_off <= t_geo + 1e-9, f"violated at R={R}"

    def test_full_ordering_holds(self):
        """t_encap <= t_coolers_off <= t_geo for a design needing real cooling."""
        props, cfg = _make_props(), _make_cfg()
        cheap = evaluate_cheap(0.25, 15.0, 100.0, props, cfg)
        gate = evaluate_gate(0.25, 15.0, _STRONG, props, cfg, cheap["t_coolers_off_yr"])

        assert gate["Feasible"]
        assert gate["t_encap_yr"] <= cheap["t_coolers_off_yr"] + 1e-9
        assert cheap["t_coolers_off_yr"] <= cheap["t_geo_yr"] + 1e-9

    def test_active_duration_is_the_difference(self):
        """t_active = t_coolers_off - t_encap whenever cooling is actually needed."""
        props, cfg = _make_props(), _make_cfg()
        cheap = evaluate_cheap(0.25, 15.0, 100.0, props, cfg)
        gate = evaluate_gate(0.25, 15.0, _STRONG, props, cfg, cheap["t_coolers_off_yr"])

        assert gate["t_active_yr"] > 0.0, "test design should need active cooling"
        assert gate["t_encap_yr"] + gate["t_active_yr"] == pytest.approx(
            cheap["t_coolers_off_yr"], rel=1e-6,
        )

    def test_no_cooling_needed_when_encapsulation_is_late(self):
        """
        If the waste is only sealed after it is already passively safe, the
        coolers never run — the LWR case.
        """
        props = _make_props()
        cfg = _make_cfg(pre_encap_min_years=8.0, pre_encap_max_years=10.0)
        cheap = evaluate_cheap(0.15, 10.0, 100.0, props, cfg)
        gate = evaluate_gate(0.15, 10.0, _WEAK, props, cfg, cheap["t_coolers_off_yr"])

        assert gate["Feasible"]
        assert gate["t_active_yr"] == 0.0


# ===========================================================================
# Monotonic responses
# ===========================================================================

class TestMonotonicity:

    def test_weaker_cooling_forces_later_encapsulation(self):
        """A technology that removes less heat needs more decay before sealing."""
        props, cfg = _make_props(), _make_cfg()
        weak = find_min_encap_years(0.25, 0.15, props, _RHO, _WEAK, cfg)
        strong = find_min_encap_years(0.25, 0.15, props, _RHO, _STRONG, cfg)
        assert strong <= weak
        assert weak > 0.0, "weak archetype should not cope immediately"

    def test_higher_loading_delays_every_milestone(self):
        """More waste per canister means more heat, so everything shifts later."""
        props, cfg = _make_props(), _make_cfg()
        low = evaluate_cheap(0.20, 5.0, 100.0, props, cfg)
        high = evaluate_cheap(0.20, 20.0, 100.0, props, cfg)

        assert high["t_coolers_off_yr"] >= low["t_coolers_off_yr"]
        assert high["t_geo_yr"] >= low["t_geo_yr"]
        assert high["N_canisters"] < low["N_canisters"]

    def test_salt_allows_earlier_emplacement_than_bentonite(self):
        """A higher surface limit means less waiting before the repository."""
        props, cfg = _make_props(), _make_cfg()
        bentonite = evaluate_cheap(0.20, 10.0, 100.0, props, cfg)
        salt = evaluate_cheap(0.20, 10.0, 200.0, props, cfg)
        assert salt["t_geo_yr"] <= bentonite["t_geo_yr"]

    def test_geology_does_not_change_coolers_off(self):
        """The interim store has no buffer, so geology is irrelevant to it."""
        props, cfg = _make_props(), _make_cfg()
        bentonite = evaluate_cheap(0.20, 10.0, 100.0, props, cfg)
        salt = evaluate_cheap(0.20, 10.0, 200.0, props, cfg)
        assert bentonite["t_coolers_off_yr"] == pytest.approx(salt["t_coolers_off_yr"])

    def test_surface_binds_at_repository_emplacement(self):
        """The buffer limit is expected to govern, not the devitrification limit."""
        props, cfg = _make_props(), _make_cfg()
        result = evaluate_cheap(0.20, 10.0, 100.0, props, cfg)
        assert result["Binding_At_Geo"] == "surface"


# ===========================================================================
# Feasibility window
# ===========================================================================

class TestFeasibilityWindow:

    def test_infeasible_when_window_closes_too_early(self):
        """A technology that cannot cope inside the user's window is rejected."""
        props = _make_props(decay_func=lambda t: 5000.0 * np.exp(-0.05 * t))
        cfg = _make_cfg(pre_encap_min_years=0.0, pre_encap_max_years=0.02)
        cheap = evaluate_cheap(0.30, 20.0, 100.0, props, cfg)
        gate = evaluate_gate(0.30, 20.0, _WEAK, props, cfg, cheap["t_coolers_off_yr"])
        assert not gate["Feasible"]

    def test_encapsulation_respects_the_earliest_delivery(self):
        """Waste cannot be sealed sooner than the operator can deliver it."""
        props = _make_props(decay_func=lambda t: 0.01 * np.exp(-0.5 * t))
        cfg = _make_cfg(pre_encap_min_years=2.0)
        cheap = evaluate_cheap(0.10, 5.0, 100.0, props, cfg)
        gate = evaluate_gate(0.10, 5.0, _STRONG, props, cfg, cheap["t_coolers_off_yr"])
        assert gate["t_encap_yr"] == pytest.approx(2.0)


# ===========================================================================
# Loading grid resolution
# ===========================================================================

class TestLoadingGrid:

    def test_explicit_list_wins_over_the_range(self):
        """A user naming two loadings must get exactly those two."""
        cfg = _make_cfg(loadings_pct=[7.0, 12.0], loadings_min=5.0,
                        loadings_max=25.0, loadings_steps=11)
        assert resolve_loadings(cfg) == [7.0, 12.0]

    def test_range_is_used_when_no_list_is_given(self):
        cfg = _make_cfg(loadings_pct=None, loadings_min=5.0,
                        loadings_max=25.0, loadings_steps=5)
        assert resolve_loadings(cfg) == [5.0, 10.0, 15.0, 20.0, 25.0]


# ===========================================================================
# End-to-end sweep
# ===========================================================================

_RADII = np.array([0.08, 0.15, 0.25])
_LOADINGS = [5.0, 15.0]
_ARCHETYPES = ["NaturalAir", "WaterPool"]


class TestRunExploration:

    @pytest.fixture(scope="class")
    def full_df(self):
        cfg = _make_cfg()
        return run_exploration(
            cfg=cfg,
            radii=_RADII,
            loadings_pct=_LOADINGS,
            archetype_names=_ARCHETYPES,
            verbose=False,
        )

    def test_sweep_is_complete(self, full_df):
        """
        Every grid point must appear for every geology and technology.  The
        output is a map, and a map with holes in it is not usable.
        """
        cfg = _make_cfg()
        expected = (
            len(_RADII) * len(_LOADINGS)
            * len(cfg["surface_limits_C"]) * len(_ARCHETYPES)
        )
        assert len(full_df) == expected

        for _, group in full_df.groupby(["Geology", "Archetype"]):
            pairs = set(zip(group["Radius_m"], group["Loading_Pct"]))
            assert len(pairs) == len(_RADII) * len(_LOADINGS)

    def test_cheap_tier_is_never_missing(self, full_df):
        """
        The passive milestones are computed unconditionally, so they must be
        present on every row even where the transient gate found no answer.
        """
        for col in ("N_canisters", "t_coolers_off_yr", "t_geo_yr",
                    "Binding_At_Geo"):
            assert full_df[col].notna().all(), col

    def test_encapsulation_does_not_depend_on_geology(self, full_df):
        """
        t_encap is solved once per (design, technology) and shared across
        geologies. If that ever stops holding, the sweep is doing duplicate
        FEM work or joining the wrong rows together.
        """
        grouped = full_df.groupby(
            ["Archetype", "Radius_m", "Loading_Pct"],
        )["t_encap_yr"]
        assert (grouped.nunique(dropna=False) <= 1).all()

    def test_milestone_arithmetic_holds_everywhere(self, full_df):
        """Every reported row must satisfy the model's own definitions."""
        feasible = full_df[full_df["Feasible"].fillna(False)]
        assert not feasible.empty

        for _, row in feasible.iterrows():
            assert row["t_encap_yr"] <= row["t_coolers_off_yr"] + 1e-6 or \
                   row["t_active_yr"] == 0.0
            assert row["t_coolers_off_yr"] <= row["t_geo_yr"] + 1e-6
            expected = max(0.0, row["t_coolers_off_yr"] - row["t_encap_yr"])
            assert row["t_active_yr"] == pytest.approx(expected, abs=1e-6)

    def test_reports_operating_conditions(self, full_df):
        """Each row must be traceable to the conditions that produced it."""
        for col in ("h_active", "T_ambient_active_C", "h_passive",
                    "T_ambient_passive_C"):
            assert col in full_df.columns
            assert full_df[col].notna().all()

    def test_facility_duty_scales_with_fleet(self, full_df):
        """Total duty is the per-canister output times the number of canisters."""
        feasible = full_df[full_df["Feasible"].fillna(False)]
        row = feasible.iloc[0]
        assert row["Facility_Duty_W"] == pytest.approx(
            row["Q_per_canister_W"] * row["N_canisters"], rel=1e-6,
        )

    def test_missing_campaign_mass_is_reported_clearly(self):
        """Canister counts are impossible without a campaign mass."""
        cfg = _make_cfg(total_waste_mass_kg=None)
        with pytest.raises(ValueError, match="total waste mass"):
            run_exploration(cfg=cfg, verbose=False)

    def test_unknown_geology_is_rejected(self):
        cfg = _make_cfg()
        with pytest.raises(ValueError, match="Unknown repository geology"):
            run_exploration(cfg=cfg, repositories=["Granite"], verbose=False)

    def test_infeasible_designs_are_kept_not_dropped(self):
        """
        A design no technology can handle still belongs in the output: the
        encapsulation map shades that region, which needs the rows present.
        """
        props = _make_props(decay_func=lambda t: 5000.0 * np.exp(-0.05 * t))
        cfg = _make_cfg(waste_form=props, pre_encap_max_years=0.02)
        result = run_exploration(
            cfg=cfg,
            radii=np.array([0.25, 0.30]),
            loadings_pct=[20.0],
            archetype_names=["NaturalAir"],
            verbose=False,
        )
        assert not result.empty
        assert not result["Feasible"].any()
        assert result["t_geo_yr"].notna().all()
