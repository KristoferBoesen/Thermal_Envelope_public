"""
Analytical steady-state solutions for the 1D cylindrical heat equation.

For a solid cylinder with uniform volumetric heat generation Q_vol [W/m³],
constant thermal conductivity k [W/(m·K)], radius R [m], and Robin BC at
the surface (HTC h [W/(m²·K)], far-field temperature T_∞ [K]):

    T(r) = T_∞ + Q·R/(2h) + Q·(R² − r²)/(4k)

Key results:
    T_center  = T_∞ + Q·R/(2h) + Q·R²/(4k)        (r = 0)
    T_surface = T_∞ + Q·R/(2h)                      (r = R)

These expressions are inverted to find the maximum allowable Q_vol that
keeps temperatures within repository safety limits.
"""

import numpy as np
from typing import Callable


def steady_state_centerline(
    Q_vol: float, R: float, k: float, h: float, T_inf: float,
) -> float:
    """
    Steady-state centreline temperature for a solid cylinder with Robin BC.

    Parameters
    ----------
    Q_vol : float
        Volumetric heat generation [W/m³].
    R : float
        Cylinder radius [m].
    k : float
        Thermal conductivity [W/(m·K)].
    h : float
        Convective HTC [W/(m²·K)].
    T_inf : float
        Far-field (ambient) temperature [K].

    Returns
    -------
    float
        Centreline temperature [K].
    """
    return T_inf + (Q_vol * R) / (2.0 * h) + (Q_vol * R**2) / (4.0 * k)


def steady_state_surface(
    Q_vol: float, R: float, h: float, T_inf: float,
) -> float:
    """
    Steady-state surface temperature for a solid cylinder with Robin BC.

    Parameters
    ----------
    Q_vol : float
        Volumetric heat generation [W/m³].
    R : float
        Cylinder radius [m].
    h : float
        Convective HTC [W/(m²·K)].
    T_inf : float
        Far-field (ambient) temperature [K].

    Returns
    -------
    float
        Surface temperature [K].
    """
    return T_inf + (Q_vol * R) / (2.0 * h)


def max_allowable_heat_rate(
    R: float,
    h: float,
    T_inf: float,
    T_limit_center: float,
    T_limit_surface: float,
    k_func: Callable,
) -> float:
    """
    Maximum allowable volumetric heat generation [W/m³] subject to both
    centreline and surface temperature limits.

    The centreline constraint depends on k, which is temperature-dependent.
    We evaluate k at the midpoint between T_∞ and the centreline limit as a
    first-order linearisation — acceptable because the passive-phase
    temperature range is narrow.

    Parameters
    ----------
    R : float
        Cylinder radius [m].
    h : float
        Heat transfer coefficient [W/(m²·K)].
    T_inf : float
        Ambient temperature [K].
    T_limit_center : float
        Max allowable centreline temperature [K].
    T_limit_surface : float
        Max allowable surface temperature [K].
    k_func : Callable
        ``k(T) → float``, thermal conductivity as a function of temperature.

    Returns
    -------
    float
        Maximum allowable Q_vol [W/m³].
    """
    # Surface constraint (independent of k)
    Q_max_surface = (T_limit_surface - T_inf) / (R / (2.0 * h))

    # Centreline constraint (depends on k)
    T_avg = 0.5 * (T_inf + T_limit_center)
    k_val = k_func(T_avg)
    k_avg = float(np.asarray(k_val).flat[0])
    Q_max_center = (T_limit_center - T_inf) / (R / (2.0 * h) + R**2 / (4.0 * k_avg))

    return min(Q_max_center, Q_max_surface)
