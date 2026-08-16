"""
Tests for the cooling technology library.

Two keys do two different jobs, and confusing them is the mistake this module
exists to prevent:

``cooling_archetypes``
    **Defines the library.**  What is written there is what exists, the same
    way ``materials`` and ``repositories`` work.  Deleting an entry removes
    that technology.

``archetypes``
    **Selects.**  Narrows what a run compares, without deleting definitions.

An earlier version merged ``cooling_archetypes`` into the built-in library
instead of replacing it, so a user who listed one technology and expected only
that one to run got all of them.  The tests below pin the replacement
semantics, the selection, and the fallback used when the block is absent.
"""

import pytest

from aethon.design.archetypes import (
    BUILTIN_ARCHETYPES,
    resolve_archetypes,
    select_archetypes,
)


def _cfg(**overrides):
    cfg = {"cooling_archetypes": {}, "archetype_names": None}
    cfg.update(overrides)
    return cfg


class TestResolveDefinitions:

    def test_bare_config_gives_the_built_ins(self):
        assert set(resolve_archetypes(_cfg())) == set(BUILTIN_ARCHETYPES)

    def test_an_override_replaces_the_figures_only(self):
        cfg = _cfg(cooling_archetypes={
            "ForcedAir": {"h": 18.0, "ambient_C": 45.0},
        })
        library = resolve_archetypes(cfg)

        assert library["ForcedAir"]["h"] == 18.0
        assert library["ForcedAir"]["ambient_C"] == 45.0
        # and the description survives, since only the figures were given
        assert library["ForcedAir"]["description"]

    def test_the_block_defines_the_library_outright(self):
        """
        What is written is what exists. An earlier version merged this into
        the built-ins, so a user who listed one technology reasonably expected
        the others to disappear and instead got a run comparing all of them.
        """
        cfg = _cfg(cooling_archetypes={
            "ForcedAir": {"h": 18.0, "ambient_C": 45.0},
        })
        assert list(resolve_archetypes(cfg)) == ["ForcedAir"]

    def test_a_wholly_new_library_is_honoured(self):
        cfg = _cfg(cooling_archetypes={
            "MyChiller": {"h": 400.0, "ambient_C": 18.0},
            "MyVault": {"h": 6.0, "ambient_C": 35.0},
        })
        assert list(resolve_archetypes(cfg)) == ["MyChiller", "MyVault"]

    def test_a_definition_without_a_description_still_works(self):
        cfg = _cfg(cooling_archetypes={
            "MyChiller": {"h": 400.0, "ambient_C": 18.0},
        })
        assert resolve_archetypes(cfg)["MyChiller"]["description"]

    def test_the_shipped_config_matches_the_python_fallback(self):
        """
        The library is written out in solver_config.yaml, and duplicated in
        Python for configs that omit the block. This catches the two drifting.
        """
        import yaml
        from pathlib import Path

        shipped = yaml.safe_load(
            (Path(__file__).parent.parent / "solver_config.yaml")
            .read_text(encoding="utf-8")
        )["cooling_archetypes"]

        assert set(shipped) == set(BUILTIN_ARCHETYPES)
        for name, spec in BUILTIN_ARCHETYPES.items():
            assert shipped[name]["h"] == spec["h"], name
            assert shipped[name]["ambient_C"] == spec["ambient_C"], name

    @pytest.mark.parametrize("spec,missing", [
        ({"h": 25.0}, "ambient_C"),
        ({"ambient_C": 40.0}, "h"),
        ({}, "h"),
    ])
    def test_an_incomplete_definition_is_rejected(self, spec, missing):
        """
        An HTC without the temperature it works against is meaningless, so a
        half-defined technology must fail rather than inherit a stray default.
        """
        cfg = _cfg(cooling_archetypes={"Broken": spec})
        with pytest.raises(ValueError, match=missing):
            resolve_archetypes(cfg)


class TestSelection:

    def test_no_selection_compares_everything(self):
        assert set(select_archetypes(_cfg())) == set(BUILTIN_ARCHETYPES)

    def test_config_list_narrows_the_run(self):
        cfg = _cfg(archetype_names=["ForcedAir"])
        assert list(select_archetypes(cfg)) == ["ForcedAir"]

    def test_explicit_names_override_the_config(self):
        """--archetype must win over the file, like every other flag."""
        cfg = _cfg(archetype_names=["ForcedAir"])
        assert list(select_archetypes(cfg, ["NaturalAir"])) == ["NaturalAir"]

    def test_selection_preserves_the_requested_order(self):
        cfg = _cfg()
        chosen = ["WaterPool", "NaturalAir"]
        assert list(select_archetypes(cfg, chosen)) == chosen

    def test_a_user_defined_technology_can_be_selected(self):
        cfg = _cfg(
            cooling_archetypes={"MyChiller": {"h": 400.0, "ambient_C": 18.0}},
            archetype_names=["MyChiller"],
        )
        assert list(select_archetypes(cfg)) == ["MyChiller"]

    def test_selection_carries_the_overridden_figures(self):
        """Define and select together, which is the realistic case."""
        cfg = _cfg(
            cooling_archetypes={"ForcedAir": {"h": 18.0, "ambient_C": 45.0}},
            archetype_names=["ForcedAir"],
        )
        selected = select_archetypes(cfg)

        assert list(selected) == ["ForcedAir"]
        assert selected["ForcedAir"]["h"] == 18.0

    def test_unknown_name_in_the_config_names_the_source(self):
        """
        The message has to say where the bad name came from - the config and
        the flag fail identically otherwise, and they are fixed in different
        places.
        """
        cfg = _cfg(archetype_names=["Cryogenic"])
        with pytest.raises(ValueError, match="config's 'archetypes' list"):
            select_archetypes(cfg)

    def test_unknown_name_on_the_flag_names_the_source(self):
        with pytest.raises(ValueError, match="--archetype"):
            select_archetypes(_cfg(), ["Cryogenic"])

    def test_unknown_name_lists_what_is_available(self):
        with pytest.raises(ValueError, match="NaturalAir"):
            select_archetypes(_cfg(), ["Cryogenic"])

    def test_missing_key_is_tolerated(self):
        """A hand-built cfg without the key must not raise."""
        assert set(select_archetypes({"cooling_archetypes": {}})) == set(
            BUILTIN_ARCHETYPES
        )
