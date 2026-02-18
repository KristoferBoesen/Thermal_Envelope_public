"""
Design envelope visualisation: vertically stacked subplot design.

Produces a figure with two subplots sharing the same X-axis (Radius):

    - **Top:**    Required Active Cooling (h) vs Radius
    - **Bottom:** Required Cooling Time (Years) vs Radius

One curve per waste-loading percentage.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional


def plot_design_envelope(
    df: pd.DataFrame,
    label: str,
    repo_type: str,
    output_dir: Path,
    loading_pct: Optional[float] = None,
    max_h: Optional[float] = None,
    max_cool: Optional[float] = None,
) -> Path:
    """
    Generate a vertically stacked design envelope plot.

    Parameters
    ----------
    df : pd.DataFrame
        Output from :func:`~src.analysis.pipeline.run_design_envelope`.
        Must contain columns ``Radius_m``, ``Loading_Pct``,
        ``Min_H_Active``, ``Min_Cooling_Years``.
    label : str
        Waste form label for the plot title and output filename.
    repo_type : str
        Repository type for the plot title and output filename.
    output_dir : Path
        Directory to write the PNG file.
    loading_pct : float, optional
        If given, plot only this loading.  Otherwise plot all loadings.
    max_h : float, optional
        Y-axis ceiling for the h subplot [W/(m²·K)].  When set, all curves
        are visually clipped at this value so every line reaches the same top
        edge.
    max_cool : float, optional
        Y-axis ceiling for the cooling-years subplot [years].  Same effect.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    fig, (ax_h, ax_t) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={"hspace": 0.12},
    )

    if loading_pct is not None:
        subset = df[df["Loading_Pct"] == loading_pct]
        groups = [(loading_pct, subset)]
    else:
        groups = list(df.groupby("Loading_Pct"))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(len(groups), 1)))

    for (lpct, grp), color in zip(groups, colors):
        grp_sorted = grp.sort_values("Radius_m")
        line_label = f"{lpct:.0f}% loading"

        # Top: h_active vs radius
        valid_h = grp_sorted.dropna(subset=["Min_H_Active"])
        if not valid_h.empty:
            ax_h.plot(
                valid_h["Radius_m"].values,
                valid_h["Min_H_Active"].values,
                "-o", color=color, label=line_label, markersize=3,
            )

        # Bottom: cooling years vs radius
        valid_t = grp_sorted.dropna(subset=["Min_Cooling_Years"])
        if not valid_t.empty:
            ax_t.plot(
                valid_t["Radius_m"].values,
                valid_t["Min_Cooling_Years"].values,
                "-s", color=color, label=line_label, markersize=3,
            )

    # Top subplot formatting
    ax_h.set_ylabel(r"Required Active Cooling, $h$ [W/(m$^2\cdot$K)]")
    ax_h.set_title(f"Design Envelope — {label} / {repo_type}")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.grid(True, alpha=0.3)
    if max_h is not None:
        ax_h.set_ylim(bottom=0, top=max_h)

    # Bottom subplot formatting
    ax_t.set_xlabel("Canister Radius [m]")
    ax_t.set_ylabel("Required Cooling Time [Years]")
    ax_t.legend(fontsize=8, loc="upper left")
    ax_t.grid(True, alpha=0.3)
    if max_cool is not None:
        ax_t.set_ylim(bottom=0, top=max_cool)

    fig.align_ylabels([ax_h, ax_t])

    filename = f"Design_Envelope_{label}_{repo_type}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath
