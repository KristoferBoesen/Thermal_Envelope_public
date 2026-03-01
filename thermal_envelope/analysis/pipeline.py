"""
Design envelope pipeline: cooling schedule computation.

For each (loading, radius) combination this module answers:

1. **Active phase** — What is the minimum convective HTC (h_active) that keeps
   T_center below the glass-transition limit during peak heat generation?

2. **Passive phase** — How many years of active cooling are required before
   the canister can safely transition to natural convection (h_passive)?

Both questions are solved via Brent's root-finding method on monotonic
residual functions.
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from typing import Dict, List, Any

from thermal_envelope.constants import KELVIN_OFFSET, H_SEARCH_MAX, T_SEARCH_MAX_YEARS
from thermal_envelope.physics.fem_solver import WasteForm
from thermal_envelope.physics.analytical import max_allowable_heat_rate


def _effective_density(rho_base: float, loading_fraction: float) -> float:
    """Effective density accounting for waste loading: ρ_base / (1 − f)."""
    return rho_base / (1.0 - loading_fraction)


def find_min_h_active(
    R: float,
    loading_fraction: float,
    properties: dict,
    rho_base: float,
    cfg: Dict[str, Any],
) -> float:
    """
    Find the minimum HTC that keeps peak centreline temperature below the
    effective safety limit.

    The effective limit is ``centerline_limit_C / safety_factor``, converted
    to Kelvin.  Uses Brent's method on the residual:

        f(h) = T_center_peak(h) − T_effective_limit

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
        Base glass density [kg/m³].
    cfg : dict
        Parsed configuration from ``solver_config.yaml``.

    Returns
    -------
    float
        Minimum h_active [W/(m²·K)].
        Returns ``np.nan`` if passive cooling already suffices.
        Returns ``np.inf`` if infeasible at the upper search bound.
    """
    T_limit_K = (cfg["centerline_limit_C"] / cfg["safety_factor"]) + KELVIN_OFFSET
    T_inf_K = cfg["ambient_temp_C"] + KELVIN_OFFSET
    eff_rho = _effective_density(rho_base, loading_fraction)
    cooling_years = cfg["cooling_months"] / 12.0

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

    h_low = cfg["h_passive"]
    if residual(h_low) <= 0.0:
        return np.nan  # passive cooling sufficient — no active h required

    if residual(H_SEARCH_MAX) > 0.0:
        return np.inf  # infeasible at any practical h

    return brentq(residual, h_low, H_SEARCH_MAX, xtol=0.01, rtol=1e-3)


def find_min_cooling_years(
    R: float,
    loading_fraction: float,
    properties: dict,
    rho_base: float,
    repo_type: str,
    cfg: Dict[str, Any],
) -> float:
    """
    Find the minimum years of active cooling before the canister can safely
    transition to passive storage.

    Uses the analytical steady-state model to compute Q_allowable under
    passive convection, then inverts the decay curve via Brent's method:

        g(t) = Q_decay(t + t_cool) · ρ_eff · loading  −  Q_allowable  =  0

    Parameters
    ----------
    R : float
        Canister radius [m].
    loading_fraction : float
        Waste loading fraction.
    properties : dict
        Waste form material properties.
    rho_base : float
        Base glass density [kg/m³].
    repo_type : str
        Repository type key (must match a key in ``surface_limits_C``).
    cfg : dict
        Parsed configuration from ``solver_config.yaml``.

    Returns
    -------
    float
        Minimum cooling time [years].
        Returns ``0.0`` if immediately safe for passive storage.
        Returns ``np.inf`` if not safe within the search window.
    """
    sf = cfg["safety_factor"]
    T_inf_K = cfg["ambient_temp_C"] + KELVIN_OFFSET
    T_limit_center_K = (cfg["centerline_limit_C"] / sf) + KELVIN_OFFSET
    T_limit_surface_K = (cfg["surface_limits_C"][repo_type] / sf) + KELVIN_OFFSET
    eff_rho = _effective_density(rho_base, loading_fraction)
    t_cool = cfg["cooling_months"] / 12.0

    Q_allowable = max_allowable_heat_rate(
        R=R,
        h=cfg["h_passive"],
        T_inf=T_inf_K,
        T_limit_center=T_limit_center_K,
        T_limit_surface=T_limit_surface_K,
        k_func=properties["k"],
    )

    decay_func = properties["decay"]

    def g(t: float) -> float:
        """Residual: actual volumetric heat generation minus allowable."""
        Q_vol_actual = decay_func(t + t_cool) * eff_rho * loading_fraction
        return Q_vol_actual - Q_allowable

    if g(0.0) <= 0.0:
        return 0.0  # immediately safe

    if g(T_SEARCH_MAX_YEARS) > 0.0:
        return np.inf  # never reaches passive safety within search window

    return brentq(g, 0.0, T_SEARCH_MAX_YEARS, xtol=0.01, rtol=1e-4)


def run_design_envelope(
    label: str,
    properties: dict,
    repo_type: str,
    loadings_pct: List[float],
    radii: np.ndarray,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute the full design envelope for one waste form and repository type.

    Iterates over all (loading, radius) combinations, calling
    :func:`find_min_h_active` and :func:`find_min_cooling_years` for each.
    Each metric is evaluated independently until its own feasibility ceiling
    is reached; the loop only exits when both ceilings have been hit.

    Parameters
    ----------
    label : str
        Waste form label (used for progress output only).
    properties : dict
        Material properties: ``rho_base``, ``decay``, ``cp``, ``k``.
    repo_type : str
        Repository type key (e.g. ``"Bentonite"`` or ``"Salt"``).
    loadings_pct : list of float
        Waste loading percentages (e.g. ``[5, 10, 15, 20]``).
    radii : np.ndarray
        Array of canister radii [m] to evaluate.
    cfg : dict
        Parsed configuration from ``solver_config.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Radius_m``, ``Loading_Pct``, ``Min_H_Active``,
        ``Min_Cooling_Years``.
    """
    rho_base = properties["rho_base"]
    max_h = cfg.get("max_h_active", np.inf)
    max_cool = cfg.get("max_cooling_years", np.inf)
    records = []
    count = 0

    for loading_pct in loadings_pct:
        frac = loading_pct / 100.0
        h_ceiling_hit = False
        t_ceiling_hit = False

        for R in radii:
            count += 1
            sys.stdout.write(
                f"\r  [{label}] {count}  "
                f"R={R:.3f} m, {loading_pct:.0f}% loading..."
            )
            sys.stdout.flush()

            # --- Compute h_min (skip if already past ceiling) ---
            if not h_ceiling_hit:
                try:
                    h_min = find_min_h_active(R, frac, properties, rho_base, cfg)
                except Exception:
                    h_min = np.nan
            else:
                h_min = np.nan

            # --- Compute t_min (skip if already past ceiling) ---
            if not t_ceiling_hit:
                try:
                    t_min = find_min_cooling_years(
                        R, frac, properties, rho_base, repo_type, cfg,
                    )
                except Exception:
                    t_min = np.nan
            else:
                t_min = np.nan

            # --- Resolve h_val; mark ceiling hit after storing actual value ---
            h_val = h_min if np.isfinite(h_min) else np.nan
            if np.isfinite(h_min) and h_min > max_h:
                h_ceiling_hit = True

            # --- Resolve t_val; 0.0 → NaN (immediately safe = N/A) ---
            if np.isfinite(t_min) and t_min > 0.0:
                t_val = t_min
            else:
                t_val = np.nan
            if np.isfinite(t_min) and t_min > max_cool:
                t_ceiling_hit = True

            records.append({
                "Radius_m": round(R, 6),
                "Loading_Pct": loading_pct,
                "Min_H_Active": h_val,
                "Min_Cooling_Years": t_val,
            })

            if h_ceiling_hit and t_ceiling_hit:
                break  # both metrics done — move to next loading

    sys.stdout.write("\r" + " " * 80 + "\r")  # clear progress line
    return pd.DataFrame(records)
