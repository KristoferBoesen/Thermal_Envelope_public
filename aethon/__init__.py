"""
aethon — AETHON: Analysis of Encapsulated Thermal Heat and Optimised Nuclides.

Thermal design envelope for vitrified nuclear waste canisters.

Quick start::

    from aethon import load_config, run_design_envelope
    import numpy as np

    cfg   = load_config("solver_config.yaml")
    radii = np.linspace(0.1, 0.5, 20)

    df = run_design_envelope(
        label       = cfg["waste_form_name"],
        properties  = cfg["waste_form"],
        repo_type   = "Bentonite",
        loadings_pct= cfg["loadings_pct"],
        radii       = radii,
        cfg         = cfg,
    )
"""

from aethon.config_loader import load_config
from aethon.analysis.pipeline import run_design_envelope
from aethon.analysis.plotting import plot_design_envelope

__all__ = [
    "load_config",
    "run_design_envelope",
    "plot_design_envelope",
]
