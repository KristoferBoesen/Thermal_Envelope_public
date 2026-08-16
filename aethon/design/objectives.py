"""
Evaluation of a single canister design.

Split into two tiers by cost, because the search relies on that split:

**Cheap** (:func:`evaluate_cheap`) — fleet size and the two passive milestones.
Closed-form geometry plus a root-find against an analytical steady-state limit;
no transient solve.  Microseconds.

**Gate** (:func:`evaluate_gate`) — the earliest encapsulation time a cooling
technology allows.  Requires repeated transient FEM solves.  Milliseconds to a
second, i.e. thousands of times more expensive.

Keeping them separate lets :mod:`aethon.design.search` sweep them independently:
the cheap tier varies with geology, the gate varies with cooling technology,
and neither varies with the other's dimension.
"""

from typing import Any, Dict, Optional

import numpy as np

from aethon.analysis.pipeline import (
    find_min_encap_years,
    find_total_decay_years,
    passive_conditions,
    peak_temperatures,
)
from aethon.constants import KELVIN_OFFSET
from aethon.design.canister import canister_count, heat_output_per_canister
from aethon.physics.analytical import allowable_heat_rate_components


def binding_constraint_at(
    R: float, surface_limit_C: float, properties: dict, cfg: Dict[str, Any],
) -> str:
    """
    Which limit sets the allowable heat rate for repository emplacement.

    Returns ``"surface"`` or ``"centre"``.  Reported rather than assumed: the
    surface limit is expected to bind for every geology, and a row saying
    otherwise is a signal worth seeing, not hiding.
    """
    sf = cfg["safety_factor"]
    T_inf_K, h_passive = passive_conditions(cfg)
    Q_centre, Q_surface = allowable_heat_rate_components(
        R=R,
        h=h_passive,
        T_inf=T_inf_K,
        T_limit_center=(cfg["centerline_limit_C"] / sf) + KELVIN_OFFSET,
        T_limit_surface=(surface_limit_C / sf) + KELVIN_OFFSET,
        k_func=properties["k"],
    )
    return "surface" if Q_surface <= Q_centre else "centre"


def evaluate_cheap(
    R: float,
    loading_pct: float,
    surface_limit_C: float,
    properties: dict,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fleet size and the two passive milestones for one design.

    Parameters
    ----------
    R : float
        Canister radius [m].
    loading_pct : float
        Waste loading [wt%].
    surface_limit_C : float
        Repository surface temperature limit [°C].
    properties : dict
        Waste form material properties.
    cfg : dict
        Parsed configuration.

    Returns
    -------
    dict
        ``N_canisters``, ``t_coolers_off_yr``, ``t_geo_yr``,
        ``Binding_At_Geo`` — times in years from reactor shutdown.
    """
    frac = loading_pct / 100.0
    rho_base = properties["rho_base"]

    n_can = canister_count(
        R=R,
        loading_fraction=frac,
        rho_base=rho_base,
        total_waste_mass_kg=cfg["total_waste_mass_kg"],
        aspect_ratio=cfg["canister_aspect_ratio"],
    )

    # Coolers can stop as soon as the centreline is passively safe. The
    # surface is unconstrained in the interim store — nothing is in contact
    # with the canister there.
    t_coolers_off = find_total_decay_years(
        R=R,
        loading_fraction=frac,
        properties=properties,
        rho_base=rho_base,
        cfg=cfg,
        surface_limit_C=np.inf,
    )

    # The repository additionally constrains the surface, via the buffer.
    t_geo = find_total_decay_years(
        R=R,
        loading_fraction=frac,
        properties=properties,
        rho_base=rho_base,
        cfg=cfg,
        surface_limit_C=surface_limit_C,
    )

    return {
        "N_canisters": n_can,
        "t_coolers_off_yr": t_coolers_off,
        "t_geo_yr": t_geo,
        "Binding_At_Geo": binding_constraint_at(R, surface_limit_C, properties, cfg),
    }


def evaluate_gate(
    R: float,
    loading_pct: float,
    archetype: Dict[str, Any],
    properties: dict,
    cfg: Dict[str, Any],
    t_coolers_off_yr: float,
) -> Dict[str, Any]:
    """
    Earliest encapsulation and resulting active-cooling duration.

    The encapsulation time is clamped into the user's acceptable
    pre-encapsulation window: it cannot be earlier than they can physically
    deliver waste, and if the archetype needs longer than their upper bound the
    design is infeasible under that technology.

    Parameters
    ----------
    R : float
        Canister radius [m].
    loading_pct : float
        Waste loading [wt%].
    archetype : dict
        Cooling technology with ``h`` and ``ambient_C``.
    properties : dict
        Waste form material properties.
    cfg : dict
        Parsed configuration.
    t_coolers_off_yr : float
        Milestone from :func:`evaluate_cheap`, reused to avoid recomputing.

    Returns
    -------
    dict
        ``t_encap_yr``, ``t_active_yr``, ``Feasible``,
        ``T_peak_centreline_C``, ``T_peak_surface_C``, ``Q_per_canister_W``.
        Facility duty is left to the caller, which knows the fleet size.
    """
    frac = loading_pct / 100.0
    rho_base = properties["rho_base"]
    t_pre_min = cfg["pre_encap_min_years"]
    t_pre_max = cfg["pre_encap_max_years"]

    t_encap = find_min_encap_years(
        R=R,
        loading_fraction=frac,
        properties=properties,
        rho_base=rho_base,
        archetype=archetype,
        cfg=cfg,
        t_max_years=t_pre_max,
    )

    # Waste cannot be encapsulated sooner than the user can deliver it.
    t_encap = max(t_encap, t_pre_min)
    feasible = bool(np.isfinite(t_encap) and t_encap <= t_pre_max)

    if not feasible:
        return {
            "t_encap_yr": np.inf,
            "t_active_yr": np.inf,
            "Feasible": False,
            "T_peak_centreline_C": np.nan,
            "T_peak_surface_C": np.nan,
            "Q_per_canister_W": np.nan,
        }

    # If the canister is already passively safe by the time it is sealed, the
    # coolers never need to run at all.
    t_active = max(0.0, t_coolers_off_yr - t_encap)

    T_centre_C, T_surface_C = peak_temperatures(
        R=R,
        loading_fraction=frac,
        properties=properties,
        rho_base=rho_base,
        h=archetype["h"],
        ambient_C=archetype["ambient_C"],
        cooling_years=t_encap,
        cfg=cfg,
    )

    Q_can = heat_output_per_canister(
        R=R,
        loading_fraction=frac,
        rho_base=rho_base,
        decay_func=properties["decay"],
        t_years=t_encap,
        aspect_ratio=cfg["canister_aspect_ratio"],
    )

    return {
        "t_encap_yr": t_encap,
        "t_active_yr": t_active,
        "Feasible": True,
        "T_peak_centreline_C": T_centre_C,
        "T_peak_surface_C": T_surface_C,
        "Q_per_canister_W": Q_can,
    }
