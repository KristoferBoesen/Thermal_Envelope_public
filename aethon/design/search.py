"""
Design-space sweep over canister radius and waste loading.

Repository geology and cooling technology are small discrete sets, enumerated
and compared side by side rather than optimised over — a user does not choose
their site's rock, and they choose a cooling system, not an HTC.

The sweep is exhaustive.  Nothing is filtered, ranked, or thinned before it
reaches the user: the whole point of the output is a map of the design space,
and a map with holes in it is worse than a slower complete one.

**Two independent passes.**  The quantities being computed depend on different
things, and running them in one nested loop made the code repeat work:

===================  ==========================  =====================
Quantity             Depends on                  Cost
===================  ==========================  =====================
``N_canisters``      R, loading                  closed form
``t_coolers_off``    R, loading                  analytic root-find
``t_geo``            R, loading, **geology**     analytic root-find
``t_encap``          R, loading, **archetype**   transient FEM
===================  ==========================  =====================

``t_encap`` has no geology dependence at all, so sweeping it inside a geology
loop solves every transient once per geology and throws all but one away.  The
two passes are run separately and joined at the end, which makes the FEM cost
proportional to ``grid x archetypes`` rather than ``grid x archetypes x
geologies``.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aethon import console
from aethon.analysis.pipeline import passive_conditions
from aethon.constants import KELVIN_OFFSET
from aethon.design.archetypes import select_archetypes
from aethon.design.objectives import evaluate_cheap, evaluate_gate


def build_grid(
    cfg: Dict[str, Any],
    radii: Optional[np.ndarray] = None,
    loadings_pct: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Radius and loading grids for the sweep.

    Radii are log-spaced: the thermal behaviour changes fastest at small radii,
    where the conduction term ``Q·R²/4k`` is still small, so a linear grid would
    waste most of its points in the flat region.  Loadings are linear — the
    effective density varies smoothly across the practical range.

    Both grids need enough points to contour against, not merely enough to
    tabulate: the output is a pair of maps over this plane.
    """
    if radii is None:
        radii = np.geomspace(
            cfg["radii_min"], cfg["radii_max"], cfg["radii_steps"],
        )
    if loadings_pct is None:
        loadings_pct = resolve_loadings(cfg)
    return np.asarray(radii, dtype=float), [float(x) for x in loadings_pct]


def resolve_loadings(cfg: Dict[str, Any]) -> List[float]:
    """
    Loading grid [wt%] from config.

    An explicit ``loadings_pct`` list wins if present — a user comparing two
    specific loadings should not have to express that as a range.  Otherwise a
    linear grid is built from ``loadings_min``/``max``/``steps``.
    """
    explicit = cfg.get("loadings_pct")
    if explicit:
        return [float(x) for x in explicit]
    return [
        float(x) for x in np.linspace(
            cfg["loadings_min"], cfg["loadings_max"], cfg["loadings_steps"],
        )
    ]


def resolve_selection(
    cfg: Dict[str, Any],
    repositories: Optional[List[str]] = None,
    archetype_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Narrow the configured geologies and cooling technologies to those requested.

    Shared with the reporting layer so that the record of a run names exactly
    what was evaluated, without re-deriving the filter and risking drift.

    Parameters
    ----------
    cfg : dict
        Parsed configuration.
    repositories : list of str, optional
        Geology names to keep.  ``None`` keeps all defined.
    archetype_names : list of str, optional
        Cooling technologies to keep.  ``None`` keeps all available.

    Returns
    -------
    tuple of (repositories, archetypes)
        Geology name -> surface limit, and the archetype specs.

    Raises
    ------
    ValueError
        If a requested name is not defined.
    """
    archetypes = select_archetypes(cfg, archetype_names)

    all_repos = cfg["surface_limits_C"]
    source = "--repo"
    if not repositories:
        repositories = cfg.get("geology_names")
        source = "the config's 'geologies' list"

    if not repositories:
        return dict(all_repos), archetypes

    unknown = [r for r in repositories if r not in all_repos]
    if unknown:
        raise ValueError(
            f"Unknown repository geology in {source}: {', '.join(unknown)}. "
            f"Available: {', '.join(all_repos)}"
        )
    return {r: all_repos[r] for r in repositories}, archetypes


def sweep_passive(
    cfg: Dict[str, Any],
    repositories: Dict[str, float],
    radii: np.ndarray,
    loadings_pct: List[float],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fleet size and the two passive milestones, over the whole grid.

    Cheap throughout — closed-form geometry plus analytic root-finds against a
    steady-state limit.  ``t_coolers_off`` carries no geology dependence, so it
    is identical across the geology blocks of the returned frame.

    Returns
    -------
    pd.DataFrame
        One row per (radius, loading, geology).
    """
    properties = cfg["waste_form"]
    rows: List[Dict[str, Any]] = []

    for repo_name, surface_limit_C in repositories.items():
        for R in radii:
            for loading_pct in loadings_pct:
                result = evaluate_cheap(
                    R=float(R),
                    loading_pct=float(loading_pct),
                    surface_limit_C=surface_limit_C,
                    properties=properties,
                    cfg=cfg,
                )
                result.update({
                    "Geology": repo_name,
                    "Radius_m": float(R),
                    "Loading_Pct": float(loading_pct),
                })
                rows.append(result)

    return pd.DataFrame(rows)


def sweep_encapsulation(
    cfg: Dict[str, Any],
    archetypes: Dict[str, Dict[str, Any]],
    radii: np.ndarray,
    loadings_pct: List[float],
    t_coolers_off: Dict[Tuple[float, float], float],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Earliest encapsulation time per cooling technology, over the whole grid.

    The expensive pass: every point is a transient FEM root-find.  Geology
    never enters, so this is swept once and shared across all of them.

    Parameters
    ----------
    t_coolers_off : dict
        ``(radius, loading_pct) -> t_coolers_off``, from the passive sweep.
        Reused rather than recomputed, and needed here because
        ``t_active = t_coolers_off − t_encap``.

    Returns
    -------
    pd.DataFrame
        One row per (radius, loading, archetype).
    """
    properties = cfg["waste_form"]
    rows: List[Dict[str, Any]] = []
    total = len(radii) * len(loadings_pct) * max(len(archetypes), 1)

    bar = (console.progress_bar("Solving encapsulation gate", total)
           if verbose else _no_progress())

    with bar as advance:
        for arch_name, archetype in archetypes.items():
            for R in radii:
                for loading_pct in loadings_pct:
                    key = (float(R), float(loading_pct))
                    gate = evaluate_gate(
                        R=float(R),
                        loading_pct=float(loading_pct),
                        archetype=archetype,
                        properties=properties,
                        cfg=cfg,
                        t_coolers_off_yr=t_coolers_off[key],
                    )
                    gate.update({
                        "Archetype": arch_name,
                        "Radius_m": float(R),
                        "Loading_Pct": float(loading_pct),
                        "h_active": float(archetype["h"]),
                        "T_ambient_active_C": float(archetype["ambient_C"]),
                    })
                    rows.append(gate)
                    advance()

    return pd.DataFrame(rows)


@contextmanager
def _no_progress():
    """Stand-in for the progress bar when the caller asked for silence."""
    yield lambda: None


def run_exploration(
    cfg: Dict[str, Any],
    repositories: Optional[List[str]] = None,
    archetype_names: Optional[List[str]] = None,
    radii: Optional[np.ndarray] = None,
    loadings_pct: Optional[List[float]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep the design space and return every evaluated point.

    Parameters
    ----------
    cfg : dict
        Parsed configuration.
    repositories : list of str, optional
        Geology names to evaluate.  ``None`` evaluates all defined.
    archetype_names : list of str, optional
        Cooling technologies to evaluate.  ``None`` evaluates all.
    radii : np.ndarray, optional
        Radius grid [m].  Defaults to a log-spaced grid from config.
    loadings_pct : list of float, optional
        Loadings [wt%].  Defaults to the config grid.
    verbose : bool
        Print progress and a summary of solver effort.

    Returns
    -------
    pd.DataFrame
        One row per (radius, loading, geology, archetype), with every
        milestone, the operating conditions behind it, and the peak
        temperatures.  Complete — no row is filtered out.
    """
    # Validate the request before touching anything expensive, so a typo in a
    # geology or technology name fails immediately with a useful message.
    repos, archetypes = resolve_selection(cfg, repositories, archetype_names)

    if cfg.get("total_waste_mass_kg") is None:
        raise ValueError(
            "No total waste mass available. Set campaign.total_waste_mass_kg "
            "in the config, pass --total-mass, or point 'waste_source' at a "
            "preprocessor output that records it."
        )

    radii, loadings_pct = build_grid(cfg, radii, loadings_pct)
    T_passive_K, h_passive = passive_conditions(cfg)

    passive_df = sweep_passive(cfg, repos, radii, loadings_pct, verbose)

    # t_coolers_off is geology-independent, so one lookup serves every block.
    t_coolers_off = {
        (row["Radius_m"], row["Loading_Pct"]): row["t_coolers_off_yr"]
        for _, row in passive_df.iterrows()
    }

    encap_df = sweep_encapsulation(
        cfg, archetypes, radii, loadings_pct, t_coolers_off, verbose,
    )

    full_df = passive_df.merge(encap_df, on=["Radius_m", "Loading_Pct"], how="left")

    full_df["Material"] = cfg["waste_form_name"]
    full_df["h_passive"] = h_passive
    full_df["T_ambient_passive_C"] = T_passive_K - KELVIN_OFFSET
    full_df["Facility_Duty_W"] = (
        full_df["Q_per_canister_W"] * full_df["N_canisters"]
    )

    full_df = full_df.sort_values(
        ["Geology", "Archetype", "Radius_m", "Loading_Pct"],
    ).reset_index(drop=True)

    return full_df
