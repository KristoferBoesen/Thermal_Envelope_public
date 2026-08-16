"""
Design-space maps: two figures, drawn over the variables the user controls.

Both figures share the same axes — canister radius against waste loading — so
they can be read against one another.  Everything else the tool computes is a
field over that plane, and the figures contour those fields directly rather
than projecting them into an objective space where the design is no longer
visible.

**Figure 1 — passive milestones.**  ``t_coolers_off`` beside ``t_geo`` for each
geology.  Read left to right it is the storage sequence: when the coolers can
stop, then when each repository will accept the canister.  ``t_coolers_off``
carries no geology dependence, so it is drawn once.

**Figure 2 — encapsulation.**  ``t_encap`` per cooling technology, with the
region no technology can hold within the user's window shaded.  Carries no
geology dependence, so one panel set covers every site.

Style follows the plots this replaces: one family of labelled isolines on
white, at round values, with no filled bands and no colourbars.  Contour levels
are chosen from a ladder of round numbers rather than by even subdivision — a
contour at 5 years is a number an engineer can act on, one at 4.7 years is not.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never displayed
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_FONT_SIZE = 12

# Round values a contour label can usefully land on, in years.  Chosen so the
# ladder stays legible across the five orders of magnitude the milestones span
# — weeks for an aggressive recycling scheme, centuries for a large canister.
_LEVEL_LADDER: Tuple[float, ...] = (
    0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0,
    50.0, 100.0, 250.0, 500.0, 1000.0,
)

# Four labelled isolines is about the most a panel carries before the labels
# start colliding in the steep corner, where the milestones change fastest.
_TARGET_LEVELS = 4

_CONTOUR_COLOR = "#0072B2"
_INFEASIBLE_COLOR = "#dcdcdc"


def _configure_style() -> None:
    """Project-standard matplotlib settings."""
    plt.rcParams.update({
        "font.size": _FONT_SIZE,
        "axes.labelsize": _FONT_SIZE,
        "xtick.labelsize": _FONT_SIZE - 1,
        "ytick.labelsize": _FONT_SIZE - 1,
        "legend.fontsize": _FONT_SIZE - 2,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 1.4,
        "savefig.dpi": 300,
        "text.usetex": False,
    })


def choose_levels(
    values: np.ndarray, target: int = _TARGET_LEVELS,
) -> List[float]:
    """
    Contour levels at round numbers spanning the data.

    Places *target* evenly log-spaced positions across the data range, then
    snaps each to the nearest rung of :data:`_LEVEL_LADDER`.  Returning round
    values matters more than hitting the target count exactly: a reader should
    be able to trace a "5 yr" line, not a "4.68 yr" one.

    Log spacing rather than linear because the milestones routinely span three
    orders of magnitude — weeks for a small canister, centuries for a large
    one — and evenly spaced levels would put every line in the same corner.

    Parameters
    ----------
    values : np.ndarray
        The field being contoured.  Non-finite entries are ignored.
    target : int
        Approximate number of levels wanted.

    Returns
    -------
    list of float
        Ascending levels, possibly empty if the data has no spread.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return []

    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return []

    inside = [x for x in _LEVEL_LADDER if lo < x < hi]

    if len(inside) > target:
        # A milestone can legitimately be zero, which has no logarithm; the
        # smallest positive rung in range is the right floor for spacing.
        floor = max(lo, inside[0] * 0.5)
        positions = np.logspace(np.log10(floor), np.log10(hi), target + 2)[1:-1]
        chosen = {
            min(inside, key=lambda rung: abs(np.log10(rung) - np.log10(p)))
            for p in positions
        }
        inside = sorted(chosen)

    if len(inside) < 2:
        # Range too narrow to contain enough rungs — subdivide it directly and
        # round to two significant figures so the labels stay readable.
        raw = np.linspace(lo, hi, target + 2)[1:-1]
        inside = sorted({float(f"{x:.2g}") for x in raw})

    return inside


def format_years(value: float) -> str:
    """Contour label for a time in years, without trailing zeros."""
    if value >= 10:
        return f"{value:.0f} yr"
    if value >= 1:
        return f"{value:g} yr"
    return f"{value:g} yr"


def pivot_field(
    df: pd.DataFrame, value_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape long-format results into a 2D field for contouring.

    Parameters
    ----------
    df : pd.DataFrame
        Rows carrying ``Radius_m``, ``Loading_Pct`` and *value_col*.  Duplicate
        (radius, loading) pairs are averaged, which is a no-op for the fields
        drawn here — they are constant across the dimension being collapsed.
    value_col : str
        Column to contour.

    Returns
    -------
    tuple of (radii, loadings, Z)
        ``Z`` has shape ``(len(loadings), len(radii))``, matching the argument
        order ``contour(radii, loadings, Z)``.  Infinities become NaN so
        matplotlib leaves them blank rather than failing.
    """
    table = df.pivot_table(
        index="Loading_Pct", columns="Radius_m", values=value_col,
        aggfunc="mean", dropna=False,
    ).sort_index().sort_index(axis=1)

    radii = table.columns.to_numpy(dtype=float)
    loadings = table.index.to_numpy(dtype=float)
    # to_numpy can hand back a read-only view of the frame's own buffer
    Z = np.array(table.to_numpy(dtype=float), copy=True)
    Z[~np.isfinite(Z)] = np.nan
    return radii, loadings, Z


def _contourable(radii: np.ndarray, loadings: np.ndarray, Z: np.ndarray) -> bool:
    """A contour needs at least a 2x2 grid with some finite spread in it."""
    if radii.size < 2 or loadings.size < 2:
        return False
    finite = Z[np.isfinite(Z)]
    return finite.size >= 2 and float(np.ptp(finite)) > 0.0


def _set_radius_ticks(ax, radii: np.ndarray) -> None:
    """
    Label the log radius axis at round values inside the swept range.

    Matplotlib's log locator puts a single decade tick on a range like
    0.05–0.5 m, which leaves the axis effectively unlabelled.
    """
    ladder = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    lo, hi = float(np.min(radii)), float(np.max(radii))
    ticks = [t for t in ladder if lo <= t <= hi]
    if len(ticks) < 2:
        return
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def _draw_panel(
    ax,
    radii: np.ndarray,
    loadings: np.ndarray,
    Z: np.ndarray,
    title: str,
    levels: Optional[Sequence[float]] = None,
    infeasible: Optional[np.ndarray] = None,
) -> bool:
    """
    Draw one field as labelled isolines.  Returns True if anything was drawn.

    ``levels`` are normally shared across every panel of a figure, so the
    panels can be read against one another — the whole reason they sit side by
    side.  Falling back to per-panel levels would make a technology that
    encapsulates twice as fast look identical to one that does not.

    ``infeasible`` is an optional boolean mask shaded grey beneath the
    contours — designs the technology cannot hold within the user's
    encapsulation window.  Shaded rather than left blank, so an empty corner
    reads as "ruled out" instead of "not computed".
    """
    ax.set_title(title, fontsize=_FONT_SIZE, pad=8)
    ax.set_xscale("log")
    _set_radius_ticks(ax, radii)

    if infeasible is not None and infeasible.any():
        ax.contourf(
            radii, loadings, infeasible.astype(float),
            levels=[0.5, 1.5], colors=[_INFEASIBLE_COLOR], zorder=0,
        )

    if not _contourable(radii, loadings, Z):
        ax.text(
            0.5, 0.5, "no variation to contour",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=_FONT_SIZE - 2, color="#888888",
        )
        return False

    if levels is None:
        levels = choose_levels(Z)
    # A level entirely outside this panel's range draws nothing, which is
    # correct and expected when levels are shared.
    levels = [v for v in levels if np.nanmin(Z) < v < np.nanmax(Z)]
    if not levels:
        return False

    contours = ax.contour(
        radii, loadings, Z, levels=levels,
        colors=_CONTOUR_COLOR, linewidths=1.4, zorder=2,
    )
    ax.clabel(
        contours, inline=True, fontsize=_FONT_SIZE - 3,
        fmt=lambda v: format_years(v),
    )
    return True


def _finish(fig, axes, xlabel: str, ylabel: str) -> None:
    """Shared axis labelling for a row of panels."""
    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()


_XLABEL = "Canister Radius (m)"
_YLABEL = "Waste Loading (wt%)"


def plot_passive_map(
    full_df: pd.DataFrame, output_dir: Path, material: str,
) -> Optional[Path]:
    """
    Figure 1: when the coolers can stop, and when each repository will accept.

    Parameters
    ----------
    full_df : pd.DataFrame
        Every evaluated design, from
        :func:`~aethon.design.search.run_exploration`.
    output_dir : Path
        Directory to write the PNG into.
    material : str
        Material name, used in the filename and title.

    Returns
    -------
    Path or None
        The figure written, or ``None`` if there was nothing to draw.
    """
    if full_df.empty:
        return None

    # One row per (geology, radius, loading): the passive milestones do not
    # vary with cooling technology, so the archetype duplication is dropped.
    passive = full_df.drop_duplicates(
        subset=["Geology", "Radius_m", "Loading_Pct"],
    )
    geologies = sorted(passive["Geology"].dropna().unique())

    # t_coolers_off is geology-independent; any single block carries it.
    first = passive[passive["Geology"] == geologies[0]] if geologies else passive
    panels = [(
        "Coolers off\n(centreline only)",
        pivot_field(first, "t_coolers_off_yr"),
    )]
    for geology in geologies:
        subset = passive[passive["Geology"] == geology]
        panels.append((
            f"Repository: {geology}\n(centreline + surface)",
            pivot_field(subset, "t_geo_yr"),
        ))

    # One level set across the row: these panels are a sequence to be read
    # against each other, and per-panel levels would hide the shift between
    # them.
    levels = choose_levels(np.concatenate([Z.ravel() for _, (_, _, Z) in panels]))

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.6 * n_panels, 4.8), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)

    drew = False
    for ax, (title, (radii, loadings, Z)) in zip(axes, panels):
        drew |= _draw_panel(ax, radii, loadings, Z, title, levels=levels)

    if not drew:
        plt.close(fig)
        return None

    fig.suptitle(
        f"{material}: passive storage milestones (years from shutdown)",
        fontsize=_FONT_SIZE + 1,
    )
    _finish(fig, axes, _XLABEL, _YLABEL)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"design_map_passive_{material}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_encapsulation_map(
    full_df: pd.DataFrame, output_dir: Path, material: str,
) -> Optional[Path]:
    """
    Figure 2: earliest encapsulation, per cooling technology.

    Shaded regions are designs the technology cannot hold below the centreline
    limit within the configured pre-encapsulation window.

    Returns
    -------
    Path or None
        The figure written, or ``None`` if there was nothing to draw.
    """
    if full_df.empty or "Archetype" not in full_df.columns:
        return None

    # t_encap does not vary with geology, so one block per archetype suffices.
    gate = full_df.drop_duplicates(
        subset=["Archetype", "Radius_m", "Loading_Pct"],
    )
    archetypes = [a for a in gate["Archetype"].dropna().unique()]
    if not archetypes:
        return None

    panels = []
    for arch in archetypes:
        subset = gate[gate["Archetype"] == arch]
        radii, loadings, Z = pivot_field(subset, "t_encap_yr")

        _, _, feasible = pivot_field(
            subset.assign(_ok=subset["Feasible"].astype(float)), "_ok",
        )
        infeasible = ~(feasible > 0.5)

        h = subset["h_active"].dropna()
        ambient = subset["T_ambient_active_C"].dropna()
        label = arch
        if not h.empty and not ambient.empty:
            label = f"{arch}\nh = {h.iloc[0]:g}, ambient {ambient.iloc[0]:g} degC"

        panels.append((label, radii, loadings, Z, infeasible))

    # Shared levels: comparing technologies is the entire purpose of this
    # figure, and it only works if the same isoline means the same thing in
    # every panel.
    levels = choose_levels(np.concatenate([Z.ravel() for _, _, _, Z, _ in panels]))

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.6 * n_panels, 4.8), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)

    drew = False
    for ax, (label, radii, loadings, Z, infeasible) in zip(axes, panels):
        drew |= _draw_panel(
            ax, radii, loadings, Z, label, levels=levels, infeasible=infeasible,
        )

    if not drew:
        plt.close(fig)
        return None

    fig.suptitle(
        f"{material}: earliest encapsulation (years from shutdown); "
        "shaded = not achievable in window",
        fontsize=_FONT_SIZE + 1,
    )
    _finish(fig, axes, _XLABEL, _YLABEL)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"design_map_encapsulation_{material}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_design_maps(
    full_df: pd.DataFrame, output_dir: Path, material: str,
) -> List[Path]:
    """
    Write both design-space maps.

    Returns
    -------
    list of Path
        Figures written, in reading order.
    """
    _configure_style()
    written = []
    for path in (
        plot_passive_map(full_df, output_dir, material),
        plot_encapsulation_map(full_df, output_dir, material),
    ):
        if path is not None:
            written.append(path)
    return written


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def sweep_stats(full_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Reduce a completed sweep to the handful of numbers worth reporting.

    Returns plain data rather than formatted text, so the arithmetic can be
    tested without a terminal and :mod:`aethon.console` owns every decision
    about how it looks.

    Parameters
    ----------
    full_df : pd.DataFrame
        Every evaluated design.

    Returns
    -------
    dict
        Empty if nothing was evaluated.  Otherwise the grid extent, one entry
        per cooling technology describing how much of the grid it can serve,
        one per geology describing the spread of ``t_geo``, and a count of
        designs where the surface limit unexpectedly did not bind.
    """
    if full_df.empty:
        return {}

    stats: Dict[str, Any] = {
        "n_designs": int(
            full_df[["Radius_m", "Loading_Pct"]].drop_duplicates().shape[0]
        ),
        "radius_min": float(full_df["Radius_m"].min()),
        "radius_max": float(full_df["Radius_m"].max()),
        "loading_min": float(full_df["Loading_Pct"].min()),
        "loading_max": float(full_df["Loading_Pct"].max()),
        "archetypes": [],
        "geologies": [],
    }

    if "Archetype" in full_df.columns:
        gate = full_df.drop_duplicates(
            subset=["Archetype", "Radius_m", "Loading_Pct"],
        )
        for arch, group in gate.groupby("Archetype", sort=False):
            feasible = group["Feasible"].fillna(False)
            n_ok = int(feasible.sum())
            total = len(group)
            stats["archetypes"].append({
                "name": str(arch),
                "n_feasible": n_ok,
                "n_total": total,
                "share_pct": 100.0 * n_ok / total if total else 0.0,
                "earliest_encap": (
                    float(group.loc[feasible, "t_encap_yr"].min())
                    if n_ok else None
                ),
            })

    passive = full_df.drop_duplicates(
        subset=["Geology", "Radius_m", "Loading_Pct"],
    )
    for geology, group in passive.groupby("Geology", sort=False):
        finite = group["t_geo_yr"].replace([np.inf, -np.inf], np.nan).dropna()
        stats["geologies"].append({
            "name": str(geology),
            "t_geo_min": None if finite.empty else float(finite.min()),
            "t_geo_max": None if finite.empty else float(finite.max()),
        })

    stats["n_unexpected_binding"] = int(
        (passive["Binding_At_Geo"] != "surface").sum()
    )
    return stats
