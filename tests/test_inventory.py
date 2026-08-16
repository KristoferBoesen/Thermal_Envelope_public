"""
Tests for deriving inventory mass from atom counts.

This replaces a mandatory user-supplied mass, so it has to be right: the whole
decay curve scales with it, and nothing downstream can detect an error.
"""

from pathlib import Path

import pandas as pd
import pytest

from decay_preprocessor.inventory import inventory_mass_kg, mass_number

_AVOGADRO = 6.02214076e23
_REAL_INVENTORY = Path(__file__).parent.parent / "examples" / "msr_inventory_5y.csv"


class TestMassNumber:

    @pytest.mark.parametrize("name,expected", [
        ("Cs137", 137),
        ("Sr90", 90),
        ("F19", 19),
        ("U235", 235),
        ("H1", 1),
        ("Ag110_m1", 110),      # metastable suffix must not win
        ("Ba137_m1", 137),
        ("Te129_m1", 129),
    ])
    def test_parses_openmc_names(self, name, expected):
        assert mass_number(name) == expected

    def test_unparseable_name_returns_zero(self):
        """A name with no digits contributes no mass rather than crashing."""
        assert mass_number("NotANuclide") == 0


class TestInventoryMass:

    def test_single_nuclide_hand_calc(self):
        """
        One mole of Cs137 weighs 137 g.

            m = (6.02214076e23 / 6.02214076e23) * 137 g = 0.137 kg
        """
        df = pd.DataFrame({"Isotope": ["Cs137"], "Atoms": [_AVOGADRO]})
        mass, unparsed = inventory_mass_kg(df)
        assert mass == pytest.approx(0.137)
        assert unparsed == []

    def test_masses_are_additive(self):
        df = pd.DataFrame({
            "Isotope": ["Cs137", "Sr90"],
            "Atoms": [_AVOGADRO, _AVOGADRO],
        })
        mass, _ = inventory_mass_kg(df)
        assert mass == pytest.approx((137 + 90) / 1000.0)

    def test_scales_linearly_with_atom_count(self):
        base = pd.DataFrame({"Isotope": ["Cs137"], "Atoms": [1e24]})
        double = pd.DataFrame({"Isotope": ["Cs137"], "Atoms": [2e24]})
        assert (inventory_mass_kg(double)[0]
                == pytest.approx(2 * inventory_mass_kg(base)[0]))

    def test_unparseable_names_are_reported_not_silent(self):
        """A name contributing no mass must be surfaced, not quietly dropped."""
        df = pd.DataFrame({
            "Isotope": ["Cs137", "Mystery"],
            "Atoms": [_AVOGADRO, _AVOGADRO],
        })
        mass, unparsed = inventory_mass_kg(df)
        assert mass == pytest.approx(0.137)
        assert unparsed == ["Mystery"]

    def test_empty_inventory_is_zero(self):
        df = pd.DataFrame({"Isotope": [], "Atoms": []})
        mass, unparsed = inventory_mass_kg(df)
        assert mass == 0.0
        assert unparsed == []

    def test_reproduces_the_shipped_inventory_header(self):
        """
        The real inventory records its own mass in a comment header, computed
        independently. Reproducing it end-to-end validates the whole method.
        """
        df = pd.read_csv(_REAL_INVENTORY, comment="#")
        mass, unparsed = inventory_mass_kg(df)

        assert unparsed == []
        assert mass == pytest.approx(115.984087, abs=1e-5)

    def test_mass_number_approximation_is_within_a_percent(self):
        """
        Using mass number instead of true atomic mass ignores the mass defect.
        That error must stay small enough to be irrelevant to the model.
        """
        true_atomic_mass = {"Cs137": 136.907, "Sr90": 89.908, "F19": 18.998}
        for name, true_u in true_atomic_mass.items():
            error = abs(mass_number(name) - true_u) / true_u
            assert error < 0.01, f"{name} off by {error:.2%}"
