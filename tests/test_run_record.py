"""
Tests for the run provenance record.

The record exists so a directory of results can be identified, and re-run, from
its own contents. Both halves of that are tested: it must contain the settings,
and feeding it back must reproduce the answer.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from aethon.config_loader import load_config
from aethon.run_record import FILENAME, build_run_record, write_run_record

_REFERENCE_CONFIG = Path(__file__).parent / "data" / "reference_config.yaml"

_REPOS = {"Salt": 200.0}
_ARCHETYPES = {"ForcedAir": {"h": 25.0, "ambient_C": 40.0}}


@pytest.fixture
def cfg():
    return load_config(_REFERENCE_CONFIG)


class TestRecordContents:

    def test_captures_settings_absent_from_the_result_csvs(self, cfg):
        """
        These are exactly the settings a results directory could not previously
        be identified by: nothing in the CSVs records them.
        """
        record = build_run_record(cfg, _REPOS, _ARCHETYPES)

        assert record["safety_factor"] == cfg["safety_factor"]
        assert record["pre_encapsulation_years"]["min"] == cfg["pre_encap_min_years"]
        assert record["pre_encapsulation_years"]["max"] == cfg["pre_encap_max_years"]
        assert record["campaign"]["total_waste_mass_kg"] == cfg["total_waste_mass_kg"]
        assert record["campaign"]["canister_aspect_ratio"] == cfg["canister_aspect_ratio"]
        assert record["decay_terms"] == cfg["decay_terms"]

    def test_is_self_contained(self, cfg):
        """
        Material coefficients and decay terms are written out, not referenced,
        so the record stays truthful if the source files are regenerated.
        """
        record = build_run_record(cfg, _REPOS, _ARCHETYPES)

        assert "waste_source" not in record
        material = record["materials"][record["material"]]
        assert "k" in material and "cp" in material and "rho_base" in material

    def test_records_only_what_was_evaluated(self, cfg):
        record = build_run_record(cfg, _REPOS, _ARCHETYPES)

        assert set(record["repositories"]) == {"Salt"}
        assert set(record["cooling_archetypes"]) == {"ForcedAir"}

    def test_grid_reflects_what_ran_not_the_config(self, cfg):
        """An overridden grid must be recorded, not the config's defaults."""
        radii = np.array([0.1, 0.2, 0.3])
        record = build_run_record(
            cfg, _REPOS, _ARCHETYPES, radii=radii, loadings_pct=[7.5],
        )

        assert record["radii"]["min"] == pytest.approx(0.1)
        assert record["radii"]["max"] == pytest.approx(0.3)
        assert record["radii"]["steps"] == 3
        assert record["loadings_pct"] == [7.5]

    def test_loading_range_is_recorded_as_the_list_it_produced(self, cfg):
        """
        A reader should not have to reproduce a linspace to know which
        loadings ran, so the range is expanded rather than copied across.
        """
        cfg = dict(cfg)
        cfg["loadings_pct"] = None
        cfg["loadings_min"], cfg["loadings_max"], cfg["loadings_steps"] = 5.0, 25.0, 5

        record = build_run_record(cfg, _REPOS, _ARCHETYPES)

        assert record["loadings_pct"] == [5.0, 10.0, 15.0, 20.0, 25.0]

    def test_candidates_are_carried_through(self, cfg):
        cfg = dict(cfg)
        cfg["candidates"] = [{"name": "A", "radius_m": 0.08, "loading_pct": 15}]

        record = build_run_record(cfg, _REPOS, _ARCHETYPES)

        assert record["candidates"][0]["name"] == "A"


class TestRoundTrip:

    def test_record_is_valid_config(self, cfg, tmp_path):
        """The whole point: the record must load straight back in."""
        write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        reloaded = load_config(tmp_path / FILENAME)

        assert reloaded["waste_form_name"] == cfg["waste_form_name"]
        assert reloaded["safety_factor"] == cfg["safety_factor"]
        assert reloaded["total_waste_mass_kg"] == cfg["total_waste_mass_kg"]
        assert set(reloaded["surface_limits_C"]) == {"Salt"}

    def test_material_expressions_survive_the_round_trip(self, cfg, tmp_path):
        """
        k and cp are expression strings. YAML must not mangle them into
        numbers or truncate them, or the reloaded material would be wrong.
        """
        write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        reloaded = load_config(tmp_path / FILENAME)

        for T in (300.0, 600.0, 900.0):
            assert reloaded["waste_form"]["k"](T) == pytest.approx(
                cfg["waste_form"]["k"](T)
            )
            assert reloaded["waste_form"]["cp"](T) == pytest.approx(
                cfg["waste_form"]["cp"](T)
            )

    def test_decay_curve_survives_the_round_trip(self, cfg, tmp_path):
        write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        reloaded = load_config(tmp_path / FILENAME)

        for t in (0.0, 1.0, 10.0):
            assert reloaded["waste_form"]["decay"](t) == pytest.approx(
                cfg["waste_form"]["decay"](t)
            )

    def test_header_reproduces_with_no_flags(self, cfg, tmp_path):
        """
        The record is self-contained, so the command needs nothing else. A
        flag here would mean something the file could not express.
        """
        path = write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        header = path.read_text(encoding="utf-8").split("\n\n")[0]

        assert "aethon --config run_config.yaml" in header
        assert "--archetype" not in header

    def test_technology_selection_survives_the_round_trip(self, cfg, tmp_path):
        """
        cooling_archetypes merges into the built-in library rather than
        replacing it, so naming entries there does not deselect the others.
        Without a separate 'archetypes' key a rerun would widen to every
        technology and produce more rows than the results it describes.
        """
        from aethon.design.archetypes import select_archetypes

        write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        reloaded = load_config(tmp_path / FILENAME)

        assert reloaded["archetype_names"] == ["ForcedAir"]
        assert list(select_archetypes(reloaded)) == ["ForcedAir"]

    def test_reselecting_a_technology_the_run_never_used_fails_loudly(
        self, cfg, tmp_path,
    ):
        """
        The record carries only the technologies that actually ran, so their
        figures are the only ones it can vouch for. Falling back to the
        built-in defaults for anything else would silently reproduce the run
        with numbers the original config may have overridden.
        """
        from aethon.design.archetypes import select_archetypes

        write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        reloaded = load_config(tmp_path / FILENAME)

        with pytest.raises(ValueError, match="Unknown cooling archetype"):
            select_archetypes(reloaded, ["NaturalAir"])

    def test_yaml_is_parseable_alongside_its_comments(self, cfg, tmp_path):
        path = write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert parsed["material"] == cfg["waste_form_name"]

    def test_candidates_survive_the_round_trip(self, cfg, tmp_path):
        """A rerun must check the same designs, not silently drop them."""
        from aethon.design.candidates import parse_candidates

        cfg = dict(cfg)
        cfg["candidates"] = [
            {"name": "A", "radius_m": 0.08, "loading_pct": 15},
            {"name": "D", "radius_m": 0.215, "loading_pct": 25},
        ]
        write_run_record(tmp_path, cfg, _REPOS, _ARCHETYPES)
        reloaded = load_config(tmp_path / FILENAME)

        parsed = parse_candidates(reloaded)
        assert [c["name"] for c in parsed] == ["A", "D"]
        assert parsed[1]["radius_m"] == pytest.approx(0.215)
