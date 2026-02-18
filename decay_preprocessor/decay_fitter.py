"""
Sum-of-exponentials fitter for decay heat curves.

Fits the model:

    Q(t) = Σᵢ  Aᵢ · exp(−λᵢ · t)

to a sampled decay heat curve using log-weighted (relative-error) residuals,
so that early-time high-power and late-time low-power behaviour are captured
with equal fractional accuracy.

The fitted ``[Amplitude, DecayConstant]`` pairs (Amplitudes in W/kg, decay
constants in yr⁻¹) can be pasted directly into ``config.yaml``.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from typing import List, Optional, Tuple


def _decay_model(t: np.ndarray, *params) -> np.ndarray:
    """
    Sum-of-exponentials decay model.

    Parameters
    ----------
    t : np.ndarray
        Time values [years].
    *params : float
        Interleaved ``[A1, λ1, A2, λ2, ...]`` where Aᵢ is the amplitude
        [W/kg] and λᵢ is the decay constant [yr⁻¹].

    Returns
    -------
    np.ndarray
        Specific decay power [W/kg].
    """
    n = len(params) // 2
    result = np.zeros_like(t, dtype=float)
    for i in range(n):
        result += params[2 * i] * np.exp(-params[2 * i + 1] * t)
    return result


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R² on the raw (non-log) data."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0


def fit_decay_curve(
    time_years: np.ndarray,
    specific_power_W_kg: np.ndarray,
    n_terms: Optional[int] = None,
    min_r2: float = 0.9999,
    max_terms: int = 6,
) -> Tuple[List[List[float]], float]:
    """
    Fit a sum-of-exponentials to a specific-power vs time curve.

    Fitting is performed with ``sigma = specific_power_W_kg`` (relative
    weighting), which is equivalent to minimising the mean squared relative
    error.  This ensures the fit is accurate across the full dynamic range of
    the decay curve rather than being dominated by the early-time peak.

    If ``n_terms`` is not specified, the function automatically tries
    3, 4, 5 … up to ``max_terms`` terms and returns the first fit that
    achieves ``min_r2``.

    Parameters
    ----------
    time_years : np.ndarray
        Evaluation times [years].  Must be positive and strictly increasing.
    specific_power_W_kg : np.ndarray
        Specific decay power [W/kg].
    n_terms : int, optional
        Fixed number of exponential terms.  If ``None``, auto-selected.
    min_r2 : float
        Minimum acceptable R² for automatic term selection (default: 0.9999).
    max_terms : int
        Maximum number of terms to attempt when auto-selecting (default: 6).

    Returns
    -------
    terms : list of [Amplitude, DecayConstant]
        Fitted parameters.  Amplitudes [W/kg] and decay constants [yr⁻¹]
        ready to paste into ``config.yaml`` as ``decay_terms``.
    r2 : float
        Coefficient of determination on the raw (non-log) data.

    Raises
    ------
    RuntimeError
        If ``curve_fit`` fails to converge for all attempted term counts.
    """
    t = np.asarray(time_years, dtype=float)
    Q = np.asarray(specific_power_W_kg, dtype=float)

    # Drop non-positive values (undefined in log-weighted residuals)
    mask = Q > 0.0
    t, Q = t[mask], Q[mask]

    Q0 = Q[0]
    t_min, t_max = t[0], t[-1]

    term_range = [n_terms] if n_terms is not None else range(3, max_terms + 1)

    best_params: Optional[np.ndarray] = None
    best_r2 = -np.inf

    for n in term_range:
        # Initial guesses: amplitudes share Q(0) equally; decay constants
        # are log-spaced to span the full observable time range.
        lam_guesses = np.geomspace(max(1.0 / t_max, 1e-4), 1.0 / t_min, n)
        p0 = []
        for lam_g in lam_guesses:
            p0 += [Q0 / n, float(lam_g)]

        bounds_lower = [0.0] * (2 * n)
        bounds_upper = [np.inf] * (2 * n)

        try:
            popt, _ = curve_fit(
                _decay_model,
                t,
                Q,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                sigma=Q,            # relative (log-space equivalent) weighting
                absolute_sigma=False,
                maxfev=100_000,
            )
        except RuntimeError:
            continue

        Q_fit = _decay_model(t, *popt)
        r2 = _r_squared(Q, Q_fit)

        if r2 > best_r2:
            best_r2 = r2
            best_params = popt

        if r2 >= min_r2:
            break

    if best_params is None:
        raise RuntimeError(
            "Sum-of-exponentials fitting failed to converge.  "
            "Try increasing --n-terms or check the input data."
        )

    n_best = len(best_params) // 2
    terms = [
        [float(best_params[2 * i]), float(best_params[2 * i + 1])]
        for i in range(n_best)
    ]
    return terms, best_r2


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
    params = [p for pair in terms for p in pair]
    Q_fit = _decay_model(t_plot, *params)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(
        time_years, specific_power_W_kg,
        "k.", markersize=2, alpha=0.5, label="Bateman solution",
    )
    ax.semilogy(
        t_plot, Q_fit,
        "r-", linewidth=2,
        label=f"Fitted ({len(terms)}-term exponential,  R² = {r2:.6f})",
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
