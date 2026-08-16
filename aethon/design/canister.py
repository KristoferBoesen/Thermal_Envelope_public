"""
Canister geometry and fleet size.

The thermal solver works per canister and in intensive units (W/kg, W/m³), so
it is indifferent to how much waste there is in total.  Fleet size is what
turns a thermal result into an engineering one: a design that runs cool because
its canisters are small and lightly loaded may need many times more of them,
with the handling, transport, and repository footprint that implies.

Canisters are modelled as solid cylinders of height ``H = aspect_ratio · R``.
Tying height to radius keeps a single decision variable — the 1D radial solver
has no axial dimension, so height only ever enters through volume.
"""

import math
from typing import Union

import numpy as np


def canister_volume(R: float, aspect_ratio: float = 6.0) -> float:
    """
    Volume of one canister [m³].

        V = π · R² · H    with    H = aspect_ratio · R

    Parameters
    ----------
    R : float
        Canister outer radius [m].
    aspect_ratio : float
        Height / radius.  The default of 6 gives H = 3 diameters.

    Returns
    -------
    float
        Canister volume [m³].
    """
    return math.pi * R**2 * (aspect_ratio * R)


def effective_density(rho_base: float, loading_fraction: float) -> float:
    """
    Density of the loaded matrix [kg/m³]: ``ρ_base / (1 − f)``.

    Mirrors the convention used throughout the solver — adding dense waste
    oxide to the matrix raises the bulk density.
    """
    return rho_base / (1.0 - loading_fraction)


def waste_mass_per_canister(
    R: float,
    loading_fraction: float,
    rho_base: float,
    aspect_ratio: float = 6.0,
) -> float:
    """
    Mass of waste (not total matrix) held by one canister [kg].

        m_waste = V · ρ_eff · f

    Parameters
    ----------
    R : float
        Canister outer radius [m].
    loading_fraction : float
        Waste loading fraction (e.g. 0.10 for 10 wt%).
    rho_base : float
        Base matrix density [kg/m³].
    aspect_ratio : float
        Height / radius.

    Returns
    -------
    float
        Waste mass per canister [kg].
    """
    volume = canister_volume(R, aspect_ratio)
    return volume * effective_density(rho_base, loading_fraction) * loading_fraction


def canister_count(
    R: float,
    loading_fraction: float,
    rho_base: float,
    total_waste_mass_kg: float,
    aspect_ratio: float = 6.0,
) -> Union[int, float]:
    """
    Number of canisters needed to encapsulate the whole campaign.

    Rounded up — a partially filled canister is still a canister.

    Parameters
    ----------
    R : float
        Canister outer radius [m].
    loading_fraction : float
        Waste loading fraction.
    rho_base : float
        Base matrix density [kg/m³].
    total_waste_mass_kg : float
        Total waste mass to encapsulate [kg].
    aspect_ratio : float
        Height / radius.

    Returns
    -------
    int
        Canister count, or ``np.nan`` if a canister would hold no waste
        (zero radius or zero loading).
    """
    per_canister = waste_mass_per_canister(
        R, loading_fraction, rho_base, aspect_ratio,
    )
    if per_canister <= 0.0:
        return np.nan
    return math.ceil(total_waste_mass_kg / per_canister)


def heat_output_per_canister(
    R: float,
    loading_fraction: float,
    rho_base: float,
    decay_func,
    t_years: float,
    aspect_ratio: float = 6.0,
) -> float:
    """
    Thermal power emitted by one canister at a given time [W].

        Q = V · ρ_eff · f · Q_specific(t)

    Used for the reported facility duty, so a user can size cooling plant
    against a wattage rather than an HTC.

    Parameters
    ----------
    R : float
        Canister outer radius [m].
    loading_fraction : float
        Waste loading fraction.
    rho_base : float
        Base matrix density [kg/m³].
    decay_func : callable
        ``Q(t_years) → specific power [W/kg]``.
    t_years : float
        Time since reactor shutdown [years].
    aspect_ratio : float
        Height / radius.

    Returns
    -------
    float
        Heat output of one canister [W].
    """
    volume = canister_volume(R, aspect_ratio)
    rho_eff = effective_density(rho_base, loading_fraction)
    return float(volume * rho_eff * loading_fraction * decay_func(t_years))
