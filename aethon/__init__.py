"""
aethon — AETHON: Analysis of Encapsulated Thermal Heat and Optimised Nuclides.

Sweeps canister designs for vitrified nuclear waste and maps the three storage
milestones over canister radius and waste loading.

Quick start::

    from aethon import load_config, run_exploration

    cfg = load_config("solver_config.yaml")

    full_df = run_exploration(
        cfg             = cfg,
        repositories    = ["Bentonite", "Salt"],
        archetype_names = None,          # None compares every technology
    )

    print(full_df[["Geology", "Archetype", "Radius_m", "Loading_Pct",
                   "N_canisters", "t_encap_yr", "t_geo_yr"]])

All times in the results are years from reactor shutdown.  See
``aethon.analysis.pipeline`` for the three storage milestones and
``aethon.design.search`` for how the sweep is structured.
"""

from aethon.config_loader import load_config
from aethon.design.archetypes import BUILTIN_ARCHETYPES, resolve_archetypes
from aethon.design.candidates import evaluate_candidates, parse_candidates
from aethon.design.report import plot_design_maps, sweep_stats
from aethon.design.search import run_exploration

__all__ = [
    "load_config",
    "run_exploration",
    "evaluate_candidates",
    "parse_candidates",
    "plot_design_maps",
    "sweep_stats",
    "resolve_archetypes",
    "BUILTIN_ARCHETYPES",
]
