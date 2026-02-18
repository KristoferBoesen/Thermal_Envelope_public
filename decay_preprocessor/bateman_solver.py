"""
Bateman equation solver for radioactive decay chains.

Solves the linear system of ODEs:

    dN/dt = A · N

where **N** is the vector of nuclide atom counts and **A** is the sparse
Bateman transmutation matrix from :func:`chain_parser.parse_chain`.

The stiff BDF integrator from ``scipy.integrate.solve_ivp`` is used with the
exact sparse Jacobian (``jac=matrix_A``) for efficiency.  Time points are
log-spaced so both early high-activity and late low-activity behaviour are
resolved with equal fractional spacing.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from typing import Dict

_EV_TO_J = 1.60218e-19       # electron-volt to Joule conversion
_SEC_PER_YEAR = 365.25 * 24 * 3600


def solve_decay(
    inventory_df: pd.DataFrame,
    nuc_to_idx: Dict[str, int],
    decay_constants: np.ndarray,
    q_values: np.ndarray,
    matrix_A: sp.csc_matrix,
    sample_mass_kg: float,
    duration_years: float,
    n_points: int = 2000,
    start_time_s: float = 1.0,
) -> pd.DataFrame:
    """
    Solve the Bateman decay equations for a given isotope inventory.

    The specific power (W/kg) at each time step is computed as:

        Q(t) = Σᵢ  Nᵢ(t) · λᵢ · Eᵢ · (1.602 × 10⁻¹⁹ J/eV)  /  m_sample

    where the sum runs over all nuclides, λᵢ is the decay constant [1/s],
    and Eᵢ is the mean energy deposited per decay [eV].

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Isotope inventory with columns ``Isotope`` (str) and ``Atoms`` (float).
    nuc_to_idx : dict
        Nuclide name → matrix index mapping from :func:`chain_parser.parse_chain`.
    decay_constants : np.ndarray, shape (N,)
        Decay constants λ [1/s].
    q_values : np.ndarray, shape (N,)
        Mean decay energy per disintegration [eV].
    matrix_A : scipy.sparse.csc_matrix, shape (N, N)
        Bateman transmutation matrix.
    sample_mass_kg : float
        Total sample mass [kg] used to normalise output to W/kg.
    duration_years : float
        Simulation duration [years].
    n_points : int
        Number of log-spaced time evaluation points.
    start_time_s : float
        First evaluation time [s]; must be > 0 for log spacing.

    Returns
    -------
    pd.DataFrame
        Columns: ``Time_Years``, ``Heat_Watts``, ``Specific_Power_W_kg``.
    """
    N = matrix_A.shape[0]

    # Build initial condition vector
    N0 = np.zeros(N, dtype=float)
    for _, row in inventory_df.iterrows():
        name = str(row["Isotope"])
        if name in nuc_to_idx:
            N0[nuc_to_idx[name]] = float(row["Atoms"])

    stop_time_s = duration_years * _SEC_PER_YEAR
    eval_times = np.geomspace(start_time_s, stop_time_s, n_points)

    sol = solve_ivp(
        fun=lambda t, y: matrix_A.dot(y),
        t_span=(0.0, stop_time_s),
        y0=N0,
        method="BDF",
        t_eval=eval_times,
        jac=matrix_A,
        rtol=1e-6,
        atol=1e-10,
    )

    heat_watts = []
    for t_idx in range(sol.y.shape[1]):
        atoms = sol.y[:, t_idx]
        activity = atoms * decay_constants          # [atoms/s] = [Bq]
        power_w = float(np.dot(activity, q_values) * _EV_TO_J)
        heat_watts.append(power_w)

    times_years = sol.t / _SEC_PER_YEAR
    specific_power = np.array(heat_watts) / sample_mass_kg

    return pd.DataFrame({
        "Time_Years": times_years,
        "Heat_Watts": heat_watts,
        "Specific_Power_W_kg": specific_power,
    })
