"""
Triple-exponential fitter for decay heat curves.

Fits the model:

    Q(t) = A1·exp(−λ1·t) + A2·exp(−λ2·t) + A3·exp(−λ3·t)

to a sampled decay heat curve using linear-space (absolute) residuals.
The first month of data is excluded to avoid fitting the early short-lived
transient, which is not relevant for canister cooling schedules on the scale
of years.

The fitted ``[Amplitude, DecayConstant]`` pairs (Amplitudes in W/kg, decay
constants in yr⁻¹) can be pasted directly into ``solver_config.yaml``.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from typing import List, Optional, Tuple


def _triple_exp(
    t: np.ndarray,
    a: float, b: float,
    c: float, d: float,
    e: float, f: float,
) -> np.ndarray:
    """
    Three-term sum-of-exponentials decay model.

    Parameters
    ----------
    t : np.ndarray
        Time values [years].
    a, c, e : float
        Amplitudes [W/kg].
    b, d, f : float
        Decay constants [yr⁻¹].

    Returns
    -------
    np.ndarray
        Specific decay power [W/kg].
    """
    return a * np.exp(-b * t) + c * np.exp(-d * t) + e * np.exp(-f * t)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R² on the raw (non-log) data."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0


def fit_decay_curve(
    time_years: np.ndarray,
    specific_power_W_kg: np.ndarray,
    cutoff_years: float = 1.0 / 12.0,
) -> Tuple[List[List[float]], float, float]:
    """
    Fit a three-term sum-of-exponentials to a specific-power vs time curve.

    Data points before ``cutoff_years`` are excluded to discard the early
    short-lived transient (dominated by nuclides with half-lives of days to
    weeks) that does not govern long-term canister cooling behaviour.

    Fitting uses linear-space (absolute) residuals so that the curve is
    optimised for accuracy across the physically relevant power range.

    Parameters
    ----------
    time_years : np.ndarray
        Evaluation times [years].  Must be positive and strictly increasing.
    specific_power_W_kg : np.ndarray
        Specific decay power [W/kg].
    cutoff_years : float
        Discard data before this time [years] (default: 1/12 ≈ 1 month).

    Returns
    -------
    terms : list of [Amplitude, DecayConstant]
        Three fitted pairs.  Amplitudes [W/kg] and decay constants [yr⁻¹]
        ready to paste into ``solver_config.yaml`` as ``decay_terms``.
    r2 : float
        Coefficient of determination on the retained (post-cutoff) data.
    rmse : float
        Root-mean-square error [W/kg] on the retained (post-cutoff) data.

    Raises
    ------
    RuntimeError
        If ``curve_fit`` fails to converge.
    """
    t = np.asarray(time_years, dtype=float)
    Q = np.asarray(specific_power_W_kg, dtype=float)

    mask = t >= cutoff_years
    t, Q = t[mask], Q[mask]

    p0 = [Q[0], 1.0, Q[0], 0.1, Q[0], 0.01]

    try:
        popt, _ = curve_fit(_triple_exp, t, Q, p0=p0, maxfev=50000)
    except RuntimeError as exc:
        raise RuntimeError(
            "Triple-exponential fitting failed to converge.  "
            "Check the input data or increase maxfev."
        ) from exc

    y_pred = _triple_exp(t, *popt)
    rmse = float(np.sqrt(np.mean((Q - y_pred) ** 2)))
    r2 = _r_squared(Q, y_pred)

    terms = [
        [float(popt[0]), float(popt[1])],
        [float(popt[2]), float(popt[3])],
        [float(popt[4]), float(popt[5])],
    ]
    return terms, r2, rmse


def plot_fit(
    time_years: np.ndarray,
    specific_power_W_kg: np.ndarray,
    terms: List[List[float]],
    r2: float,
    output_path: Optional[Path] = None,
) -> None:
    """
    Diagnostic plot: Bateman solution vs fitted sum-of-exponentials.

    Both curves are plotted on a log-y axis.  The fitted parameters and R²
    are annotated on the figure.

    Parameters
    ----------
    time_years : np.ndarray
        Evaluation times [years].
    specific_power_W_kg : np.ndarray
        Raw Bateman specific power [W/kg].
    terms : list of [Amplitude, DecayConstant]
        Fitted parameters from :func:`fit_decay_curve`.
    r2 : float
        Coefficient of determination (annotated on plot).
    output_path : Path, optional
        Save figure here if given; otherwise display interactively.
    """
    t_pos = time_years[time_years > 0]
    t_plot = np.geomspace(t_pos[0], t_pos[-1], 500)
    a, b, c, d, e, f = (p for pair in terms for p in pair)
    Q_fit = _triple_exp(t_plot, a, b, c, d, e, f)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(
        time_years, specific_power_W_kg,
        "k.", markersize=2, alpha=0.5, label="Bateman solution",
    )
    ax.semilogy(
        t_plot, Q_fit,
        "r-", linewidth=2,
        label=f"Fitted (3-term exponential,  R² = {r2:.6f})",
    )

    param_lines = "\n".join(
        f"  A{i + 1} = {A:.4g} W/kg,   λ{i + 1} = {lam:.4g} yr⁻¹"
        for i, (A, lam) in enumerate(terms)
    )
    ax.text(
        0.02, 0.05, param_lines,
        transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Specific Decay Power [W/kg]")
    ax.set_title("Decay Heat Curve — Bateman Solution vs Fitted Exponential")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
