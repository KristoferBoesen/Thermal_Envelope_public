"""
Storage milestones: when a canister may move between thermal environments.

The storage life of a canister is described by three milestones, all measured
in years **from reactor shutdown**:

``t_encap``
    Earliest the waste can be sealed into a canister and placed in the active
    cooling facility.  Gated by the peak *centreline* temperature under the
    chosen cooling archetype's HTC and ambient temperature.

``t_coolers_off``
    Earliest the active coolers can be switched off and the canister moved to a
    passive interim store.  Gated by the steady-state *centreline* temperature
    under passive conditions.  The surface temperature is unconstrained here —
    the interim store has no buffer material in contact with the canister.

``t_geo``
    Earliest the canister can be emplaced in the geological repository.  Gated
    by *both* centreline and surface temperature under passive conditions, with
    a geology-specific surface limit.

The duration the coolers must actually run is ``t_coolers_off − t_encap``.
Splitting ``t_coolers_off`` from ``t_geo`` is what keeps the active facility
from running through the whole surface-temperature wait.

Ordering ``t_encap ≤ t_coolers_off ≤ t_geo`` holds by construction: the
centreline-only allowable heat rate is always ≥ ``min(centre, surface)``.

All gates are solved via Brent's method on monotonic residual functions.  Only
``t_encap`` needs the transient FEM solver; the other two invert the decay
curve against a closed-form steady-state limit and are cheap.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, Any, Optional, Tuple

from aethon.constants import (
    H_SEARCH_MAX,
    KELVIN_OFFSET,
    MIN_H_MARGIN_C,
    T_SEARCH_MAX_YEARS,
)
from aethon.physics.fem_solver import WasteForm
from aethon.physics.analytical import max_allowable_heat_rate


def _effective_density(rho_base: float, loading_fraction: float) -> float:
    """Effective density accounting for waste loading: ρ_base / (1 − f)."""
    return rho_base / (1.0 - loading_fraction)


def passive_conditions(cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Ambient temperature [K] and HTC [W/(m²·K)] for the passive phases.

    Reads the ``passive`` block, tolerating configs that still use the older
    flat ``ambient_temp_C`` / ``h_passive`` keys.
    """
    ambient_C = cfg.get("passive_ambient_C", cfg.get("ambient_temp_C"))
    h = cfg.get("passive_h", cfg.get("h_passive"))
    if ambient_C is None or h is None:
        raise KeyError(
            "Passive conditions missing. Define a 'passive' block with "
            "'ambient_C' and 'h' in solver_config.yaml."
        )
    return ambient_C + KELVIN_OFFSET, h


def _centerline_limit_K(cfg: Dict[str, Any], margin_C: float = 0.0) -> float:
    """
    Effective centreline limit [K] after applying the safety factor.

    ``margin_C`` is subtracted from the material limit *before* the safety
    factor divides, so a run with extra margin tightens the target rather than
    leaving it fixed.
    """
    limit_C = cfg["centerline_limit_C"] - margin_C
    return (limit_C / cfg["safety_factor"]) + KELVIN_OFFSET


def peak_temperatures(
    R: float,
    loading_fraction: float,
    properties: dict,
    rho_base: float,
    h: float,
    ambient_C: float,
    cooling_years: float,
    cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Peak centreline and coincident surface temperature [°C] for one transient.

    Thin wrapper over :meth:`WasteForm.solve_for_peak` used both by the
    ``t_encap`` root-find and by the reporting layer, so the numbers shown to
    the user come from exactly the same solve that gated feasibility.

    Parameters
    ----------
    R : float
        Canister radius [m].
    loading_fraction : float
        Waste loading fraction.
    properties : dict
        Waste form material properties (``decay``, ``cp``, ``k`` callables).
    rho_base : float
        Base matrix density [kg/m³].
    h : float
        Convective HTC [W/(m²·K)].
    ambient_C : float
        Facility ambient temperature [°C].
    cooling_years : float
        Decay time already elapsed at the start of this phase [years].
    cfg : dict
        Parsed configuration.

    Returns
    -------
    tuple of (T_center_C, T_surface_C)
    """
    sim = WasteForm(
        R=R,
        ambient_T=ambient_C + KELVIN_OFFSET,
        h_coeff=h,
        loading_fraction=loading_fraction,
        properties=properties,
        cooling_years=cooling_years,
        effective_density=_effective_density(rho_base, loading_fraction),
        n_nodes=cfg["nodes"],
    )
    _, T_center_K, T_surface_K = sim.solve_for_peak(max_years=cfg["max_years"])
    return T_center_K - KELVIN_OFFSET, T_surface_K - KELVIN_OFFSET


def find_min_h_active(
    R: float,
    loading_fraction: float,
    properties: dict,
    rho_base: float,
    cfg: Dict[str, Any],
    ambient_C: float,
    cooling_years: float,
) -> float:
    """
    Minimum HTC keeping the peak centreline temperature within the limit.

    Reported alongside each frontier design as a diagnostic: if none of the
    named cooling technologies matches your facility, this is the convective
    performance you would actually need to specify.  It is only meaningful
    together with the ambient temperature it was computed at — the two trade
    off directly, since convective flux depends on ``h·(T_surface − T_ambient)``.

    The target is held ``MIN_H_MARGIN_C`` below the material limit before the
    safety factor divides, i.e. ``(centerline_limit_C − 1) / safety_factor``.
    Rooting against the limit itself would return the *critical* coefficient —
    the h at which the design sits exactly on the limit and any deviation puts
    it over — which is not a number a facility can be specified against.

    Uses Brent's method on the residual:

        f(h) = T_center_peak(h) − T_target

    which is monotonically decreasing in h.

    Parameters
    ----------
    R : float
        Canister radius [m].
    loading_fraction : float
        Waste loading fraction (e.g. 0.05 for 5 %).
    properties : dict
        Waste form material properties (``decay``, ``cp``, ``k`` callables).
    rho_base : float
        Base matrix density [kg/m³].
    cfg : dict
        Parsed configuration from ``solver_config.yaml``.
    ambient_C : float
        Facility ambient temperature [°C] the HTC is quoted against.
    cooling_years : float
        Decay time already elapsed when the canister enters the facility
        [years from shutdown].

    Returns
    -------
    float
        Minimum h_active [W/(m²·K)].
        Returns ``np.nan`` if passive cooling already suffices.
        Returns ``np.inf`` if infeasible at the upper search bound.
    """
    T_limit_K = _centerline_limit_K(cfg, margin_C=MIN_H_MARGIN_C)
    T_inf_K = ambient_C + KELVIN_OFFSET
    eff_rho = _effective_density(rho_base, loading_fraction)

    def residual(h: float) -> float:
        sim = WasteForm(
            R=R,
            ambient_T=T_inf_K,
            h_coeff=h,
            loading_fraction=loading_fraction,
            properties=properties,
            cooling_years=cooling_years,
            effective_density=eff_rho,
            n_nodes=cfg["nodes"],
        )
        _, T_center_K, _ = sim.solve_for_peak(max_years=cfg["max_years"])
        return T_center_K - T_limit_K

    _, h_low = passive_conditions(cfg)
    if residual(h_low) <= 0.0:
        return np.nan  # passive cooling sufficient — no active h required

    if residual(H_SEARCH_MAX) > 0.0:
        return np.inf  # infeasible at any practical h

    return brentq(residual, h_low, H_SEARCH_MAX, xtol=0.01, rtol=1e-3)


def find_total_decay_years(
    R: float,
    loading_fraction: float,
    properties: dict,
    rho_base: float,
    cfg: Dict[str, Any],
    surface_limit_C: float,
) -> float:
    """
    Total decay time from reactor shutdown before the canister is passively safe.

    Uses the analytical steady-state model to compute Q_allowable under passive
    convection, then inverts the decay curve via Brent's method:

        g(t) = Q_decay(t) · ρ_eff · loading  −  Q_allowable  =  0

    The returned time is absolute (measured from shutdown), not relative to any
    pre-encapsulation period — the decay curve depends only on total elapsed
    time, so every milestone can be expressed on the same clock.

    Parameters
    ----------
    R : float
        Canister radius [m].
    loading_fraction : float
        Waste loading fraction.
    properties : dict
        Waste form material properties.
    rho_base : float
        Base matrix density [kg/m³].
    cfg : dict
        Parsed configuration from ``solver_config.yaml``.
    surface_limit_C : float
        Surface temperature limit [°C].  Pass ``np.inf`` to constrain the
        centreline alone — this yields ``t_coolers_off``.  Pass a geology's
        buffer limit to yield ``t_geo``.

    Returns
    -------
    float
        Total decay time [years from shutdown].
        Returns ``0.0`` if safe immediately at shutdown.
        Returns ``np.inf`` if never safe within the search window.
    """
    sf = cfg["safety_factor"]
    T_inf_K, h_passive = passive_conditions(cfg)
    T_limit_center_K = _centerline_limit_K(cfg)
    # np.inf / sf is still np.inf, so the centreline-only case survives this
    T_limit_surface_K = (surface_limit_C / sf) + KELVIN_OFFSET
    eff_rho = _effective_density(rho_base, loading_fraction)

    Q_allowable = max_allowable_heat_rate(
        R=R,
        h=h_passive,
        T_inf=T_inf_K,
        T_limit_center=T_limit_center_K,
        T_limit_surface=T_limit_surface_K,
        k_func=properties["k"],
    )

    decay_func = properties["decay"]

    def g(t: float) -> float:
        """Residual: actual volumetric heat generation minus allowable."""
        Q_vol_actual = decay_func(t) * eff_rho * loading_fraction
        return Q_vol_actual - Q_allowable

    if g(0.0) <= 0.0:
        return 0.0  # safe from the outset

    if g(T_SEARCH_MAX_YEARS) > 0.0:
        return np.inf  # never reaches passive safety within search window

    # Tight absolute tolerance (~1 hour). This gate is analytic and costs
    # microseconds, so there is nothing to buy by loosening it — and the
    # results are contoured, where a few days of jitter between neighbouring
    # grid points shows up as a visible kink in an otherwise smooth isoline.
    return brentq(g, 0.0, T_SEARCH_MAX_YEARS, xtol=1e-4, rtol=1e-8)


def find_min_encap_years(
    R: float,
    loading_fraction: float,
    properties: dict,
    rho_base: float,
    archetype: Dict[str, Any],
    cfg: Dict[str, Any],
    t_max_years: Optional[float] = None,
) -> float:
    """
    Earliest encapsulation time [years from shutdown] for a cooling archetype.

    The canister enters the active facility after ``t_encap`` years of decay.
    Longer pre-encapsulation cooling means less decay heat at entry, so the
    peak centreline temperature is monotonically decreasing in ``t_encap`` and
    Brent's method has a unique root on:

        f(t_encap) = T_center_peak(t_encap) − T_limit

    Parameters
    ----------
    R : float
        Canister radius [m].
    loading_fraction : float
        Waste loading fraction.
    properties : dict
        Waste form material properties.
    rho_base : float
        Base matrix density [kg/m³].
    archetype : dict
        Cooling technology with keys ``h`` [W/(m²·K)] and ``ambient_C`` [°C].
    cfg : dict
        Parsed configuration from ``solver_config.yaml``.
    t_max_years : float, optional
        Upper search bound [years].  Defaults to ``cfg["pre_encap_max_years"]``
        if present, otherwise the global cooling-time search bound.  Anything
        beyond the user's acceptable pre-encapsulation window is infeasible in
        practice, so bounding here avoids wasted transient solves.

    Returns
    -------
    float
        Earliest feasible encapsulation time [years from shutdown].
        Returns ``0.0`` if the archetype can handle the waste immediately.
        Returns ``np.inf`` if it cannot within ``t_max_years``.
    """
    if t_max_years is None:
        t_max_years = cfg.get("pre_encap_max_years", T_SEARCH_MAX_YEARS)

    T_limit_K = _centerline_limit_K(cfg)

    def residual(t_encap: float) -> float:
        T_center_C, _ = peak_temperatures(
            R=R,
            loading_fraction=loading_fraction,
            properties=properties,
            rho_base=rho_base,
            h=archetype["h"],
            ambient_C=archetype["ambient_C"],
            cooling_years=t_encap,
            cfg=cfg,
        )
        return (T_center_C + KELVIN_OFFSET) - T_limit_K

    if residual(0.0) <= 0.0:
        return 0.0  # feasible straight out of the reactor

    if residual(t_max_years) > 0.0:
        return np.inf  # archetype cannot cool this design in the allowed window

    # Tight absolute tolerance (~1 hour). Encapsulation times are often only
    # weeks, so a coarse tolerance would be a large relative error — and the
    # error direction matters: converging early means sealing the waste while
    # it still exceeds the limit.
    return brentq(residual, 0.0, t_max_years, xtol=1e-4, rtol=1e-6)
