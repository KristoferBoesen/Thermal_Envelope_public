"""
Mass of an isotope inventory, computed from its own atom counts.

The Bateman solve produces watts; the solver needs watts per kilogram. That
division used to require the user to supply a mass, which was the most damaging
input in the whole toolchain: nothing downstream could detect a wrong value, so
a mistake produced plausible results that were wrong by a constant factor.

The mass is derivable from the inventory itself, so it is derived here instead.
An atom count and a nuclide name give a mass directly, and because both come
from the same file the mass cannot disagree with the heat.

    m = sum_i (N_i / N_A) * A_i

Mass number ``A`` stands in for the true atomic mass. The difference is the
nuclear binding energy, about 0.1 % (Cs-137 is 136.907 u against A = 137), which
is negligible beside every other uncertainty in the model.

This is only the *whole* waste mass if the inventory lists every nuclide
present, stable ones included. An inventory of radioactive species alone gives
a mass that is too low and therefore a specific power that is too high, which is
why the computed value is always reported rather than used silently.
"""

import re
from typing import Tuple

import pandas as pd

_AVOGADRO = 6.02214076e23

# First run of digits in an OpenMC nuclide name: "Cs137" -> 137,
# "Ag110_m1" -> 110, "U235" -> 235. Element symbols never contain digits.
_MASS_NUMBER = re.compile(r"(\d+)")


def mass_number(nuclide: str) -> int:
    """
    Mass number parsed from an OpenMC nuclide name.

    Parameters
    ----------
    nuclide : str
        Name such as ``"Cs137"`` or ``"Ag110_m1"``.

    Returns
    -------
    int
        The mass number, or 0 if the name contains no digits.
    """
    match = _MASS_NUMBER.search(nuclide)
    return int(match.group(1)) if match else 0


def inventory_mass_kg(inventory_df: pd.DataFrame) -> Tuple[float, list]:
    """
    Total mass of the inventory [kg], and any nuclides that could not be parsed.

    Every row contributes, whether or not it matched the decay chain: an
    unmatched nuclide is still mass sitting in the canister. The preprocessor
    separately reports how many nuclides matched, so a naming problem shows up
    there rather than being hidden in the mass.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Must have ``Isotope`` and ``Atoms`` columns.

    Returns
    -------
    tuple of (mass_kg, unparsed)
        ``unparsed`` lists nuclide names with no mass number, which contribute
        nothing to the total.
    """
    numbers = inventory_df["Isotope"].astype(str).apply(mass_number)
    unparsed = sorted(
        inventory_df["Isotope"][numbers == 0].astype(str).unique().tolist()
    )

    grams = (inventory_df["Atoms"] / _AVOGADRO * numbers).sum()
    return float(grams) / 1000.0, unparsed
