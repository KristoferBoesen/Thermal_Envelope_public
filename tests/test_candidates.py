"""
Tests for named candidate designs.

Candidates are the decision half of the output: exact numbers for designs
somebody has actually proposed.  Two properties matter most — a candidate
sitting on a grid point must reproduce the sweep's answer for it, and a
malformed entry must fail loudly rather than vanish.
"""

import numpy as np
import pytest

from aethon.design.candidates import evaluate_candidates, parse_candidates
from aethon.design.search import run_exploration

_RHO = 2500.0
_REPOS = {"Salt": 200.0}
_ARCHETYPES = {"ForcedAir": {"h": 25.0, "ambient_C": 40.0}}


def _make_props():
    return {
        "rho_base": _RHO,
        "decay": lambda t: 200.0 * np.exp(-0.5 * t),
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
        "candidates": [],
    }
    cfg.update(overrides)
    return cfg


class TestParseCandidates:

    def test_no_block_yields_nothing(self):
        assert parse_candidates(_make_cfg()) == []

    def test_names_default_when_omitted(self):
        cfg = _make_cfg(candidates=[{"radius_m": 0.1, "loading_pct": 10}])
        assert parse_candidates(cfg)[0]["name"] == "C1"

    def test_values_are_coerced_to_float(self):
        cfg = _make_cfg(candidates=[{"name": "A", "radius_m": 1, "loading_pct": 10}])
        parsed = parse_candidates(cfg)[0]
        assert isinstance(parsed["radius_m"], float)
        assert isinstance(parsed["loading_pct"], float)

    def test_missing_key_is_rejected(self):
        """A silently skipped candidate is indistinguishable from a dull one."""
        cfg = _make_cfg(candidates=[{"name": "A", "radius_m": 0.1}])
        with pytest.raises(ValueError, match="loading_pct"):
            parse_candidates(cfg)

    def test_non_positive_radius_is_rejected(self):
        cfg = _make_cfg(candidates=[{"name": "A", "radius_m": 0.0, "loading_pct": 10}])
        with pytest.raises(ValueError, match="positive"):
            parse_candidates(cfg)

    @pytest.mark.parametrize("loading", [0.0, 100.0, 150.0, -5.0])
    def test_out_of_range_loading_is_rejected(self, loading):
        cfg = _make_cfg(
            candidates=[{"name": "A", "radius_m": 0.1, "loading_pct": loading}],
        )
        with pytest.raises(ValueError, match="between 0 and 100"):
            parse_candidates(cfg)

    def test_non_mapping_entry_is_rejected(self):
        cfg = _make_cfg(candidates=["0.1, 10"])
        with pytest.raises(ValueError, match="not a mapping"):
            parse_candidates(cfg)


class TestEvaluateCandidates:

    def test_no_candidates_gives_an_empty_frame(self):
        assert evaluate_candidates(_make_cfg(), _REPOS, _ARCHETYPES).empty

    def test_one_row_per_combination(self):
        cfg = _make_cfg(candidates=[
            {"name": "A", "radius_m": 0.10, "loading_pct": 10},
            {"name": "B", "radius_m": 0.20, "loading_pct": 15},
        ])
        repos = {"Bentonite": 100.0, "Salt": 200.0}
        archetypes = {
            "NaturalAir": {"h": 5.0, "ambient_C": 40.0},
            "ForcedAir": {"h": 25.0, "ambient_C": 40.0},
        }
        result = evaluate_candidates(cfg, repos, archetypes)
        assert len(result) == 2 * 2 * 2
        assert set(result["Name"]) == {"A", "B"}

    def test_reproduces_the_sweep_at_a_shared_point(self):
        """
        A candidate on a grid point must agree with the sweep. If these ever
        diverge, one of the two paths is not evaluating what it claims to.
        """
        R, loading = 0.15, 10.0
        cfg = _make_cfg(
            candidates=[{"name": "X", "radius_m": R, "loading_pct": loading}],
        )
        sweep = run_exploration(
            cfg=cfg,
            radii=np.array([R]),
            loadings_pct=[loading],
            repositories=["Salt"],
            archetype_names=["ForcedAir"],
            verbose=False,
        )
        cand = evaluate_candidates(cfg, _REPOS, _ARCHETYPES)

        assert len(sweep) == 1 and len(cand) == 1
        for col in ("N_canisters", "t_encap_yr", "t_coolers_off_yr",
                    "t_active_yr", "t_geo_yr", "Facility_Duty_W"):
            assert cand.iloc[0][col] == pytest.approx(
                sweep.iloc[0][col], rel=1e-9,
            ), col

    def test_off_grid_radius_is_evaluated(self):
        """Real proposals rarely land on a geomspace point."""
        cfg = _make_cfg(
            candidates=[{"name": "Odd", "radius_m": 0.1234, "loading_pct": 12.7}],
        )
        result = evaluate_candidates(cfg, _REPOS, _ARCHETYPES)
        assert len(result) == 1
        assert result.iloc[0]["Radius_m"] == pytest.approx(0.1234)
        assert np.isfinite(result.iloc[0]["t_geo_yr"])

    def test_milestone_ordering_holds(self):
        cfg = _make_cfg(candidates=[
            {"name": "A", "radius_m": 0.10, "loading_pct": 10},
            {"name": "B", "radius_m": 0.25, "loading_pct": 20},
        ])
        result = evaluate_candidates(cfg, _REPOS, _ARCHETYPES)
        feasible = result[result["Feasible"]]
        assert not feasible.empty

        for _, row in feasible.iterrows():
            assert row["t_encap_yr"] <= row["t_coolers_off_yr"] + 1e-6
            assert row["t_coolers_off_yr"] <= row["t_geo_yr"] + 1e-6

    def test_min_h_is_quoted_with_its_ambient(self):
        """An HTC without the temperature it works against is meaningless."""
        cfg = _make_cfg(
            candidates=[{"name": "A", "radius_m": 0.20, "loading_pct": 15}],
        )
        result = evaluate_candidates(cfg, _REPOS, _ARCHETYPES)
        assert "Min_H_Active" in result.columns
        assert result["T_ambient_active_C"].notna().all()

    def test_min_h_clears_the_limit_with_margin(self):
        """
        Min_H_Active targets a degree below the limit, so applying it must
        leave the design strictly under - not exactly on - the limit.
        """
        from aethon.analysis.pipeline import peak_temperatures

        cfg = _make_cfg(
            candidates=[{"name": "A", "radius_m": 0.20, "loading_pct": 15}],
        )
        row = evaluate_candidates(cfg, _REPOS, _ARCHETYPES).iloc[0]
        if not np.isfinite(row["Min_H_Active"]):
            pytest.skip("passive cooling already suffices for this design")

        T_centre, _ = peak_temperatures(
            R=row["Radius_m"],
            loading_fraction=row["Loading_Pct"] / 100.0,
            properties=cfg["waste_form"],
            rho_base=_RHO,
            h=row["Min_H_Active"],
            ambient_C=row["T_ambient_active_C"],
            cooling_years=row["t_encap_yr"],
            cfg=cfg,
        )
        assert T_centre < cfg["centerline_limit_C"]

    def test_infeasible_candidate_is_reported_not_dropped(self):
        """A design that cannot be cooled is an answer, not an omission."""
        cfg = _make_cfg(
            pre_encap_max_years=0.01,
            candidates=[{"name": "Hot", "radius_m": 0.30, "loading_pct": 25}],
        )
        result = evaluate_candidates(
            cfg, _REPOS, {"NaturalAir": {"h": 5.0, "ambient_C": 40.0}},
        )
        assert len(result) == 1
        assert not result.iloc[0]["Feasible"]
        assert np.isfinite(result.iloc[0]["t_geo_yr"])
