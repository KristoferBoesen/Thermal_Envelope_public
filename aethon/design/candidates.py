"""
Named candidate designs: the decision half of the output.

The maps show the whole design space, which is what you want when you do not
yet know what you are looking for.  Once you do — once there are three
canisters on the table and a choice to justify — you need exact numbers for
those three, not a field to read off by eye.

That is what this module produces.  A candidate is a radius and a loading with
a name attached, listed in the config:

.. code-block:: yaml

   candidates:
     - {name: A, radius_m: 0.080, loading_pct: 15}
     - {name: D, radius_m: 0.215, loading_pct: 25}

Each is evaluated against every selected geology and cooling technology, and
reported with the full timeline rather than a single figure of merit.  Nothing
is ranked: which of them is best depends on things the solver does not know.

Candidates are deliberately **not** required to lie on the swept grid.  The
point is to check a design somebody has proposed, which will rarely coincide
with a geomspace point.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aethon.analysis.pipeline import find_min_h_active, passive_conditions
from aethon.constants import KELVIN_OFFSET
from aethon.design.objectives import evaluate_cheap, evaluate_gate

_REQUIRED = ("radius_m", "loading_pct")


def parse_candidates(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Read and validate the ``candidates`` block.

    Parameters
    ----------
    cfg : dict
        Parsed configuration.

    Returns
    -------
    list of dict
        ``{"name", "radius_m", "loading_pct"}``, empty if none are defined.

    Raises
    ------
    ValueError
        If an entry is missing a required key or carries an unusable value.
        Failing loudly matters here — a silently skipped candidate looks
        exactly like one that was evaluated and found unremarkable.
    """
    raw = cfg.get("candidates") or []
    parsed: List[Dict[str, Any]] = []

    for i, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Candidate {i} is not a mapping. Each entry needs "
                "'radius_m' and 'loading_pct', e.g. "
                "{name: A, radius_m: 0.08, loading_pct: 15}"
            )
        missing = [k for k in _REQUIRED if entry.get(k) is None]
        if missing:
            raise ValueError(
                f"Candidate {entry.get('name', i)} is missing "
                f"{', '.join(missing)}."
            )

        radius = float(entry["radius_m"])
        loading = float(entry["loading_pct"])
        if radius <= 0.0:
            raise ValueError(
                f"Candidate {entry.get('name', i)}: radius_m must be "
                f"positive, got {radius:g}."
            )
        if not 0.0 < loading < 100.0:
            raise ValueError(
                f"Candidate {entry.get('name', i)}: loading_pct must be "
                f"between 0 and 100, got {loading:g}."
            )

        parsed.append({
            "name": str(entry.get("name", f"C{i}")),
            "radius_m": radius,
            "loading_pct": loading,
        })

    return parsed


def evaluate_candidates(
    cfg: Dict[str, Any],
    repositories: Dict[str, float],
    archetypes: Dict[str, Dict[str, Any]],
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Evaluate each named design against every geology and cooling technology.

    Parameters
    ----------
    cfg : dict
        Parsed configuration.
    repositories : dict
        Geology name -> surface limit [degC], as selected for this run.
    archetypes : dict
        Cooling technologies, as selected for this run.
    candidates : list of dict, optional
        Parsed candidates.  Read from *cfg* if omitted.

    Returns
    -------
    pd.DataFrame
        One row per (candidate, geology, archetype), carrying all three
        milestones, fleet size, heat output, facility duty, and the minimum
        HTC the design would need.  Empty if no candidates are defined.
    """
    if candidates is None:
        candidates = parse_candidates(cfg)
    if not candidates:
        return pd.DataFrame()

    properties = cfg["waste_form"]
    rho_base = properties["rho_base"]
    T_passive_K, h_passive = passive_conditions(cfg)

    rows: List[Dict[str, Any]] = []

    for cand in candidates:
        R = cand["radius_m"]
        loading_pct = cand["loading_pct"]

        for geology, surface_limit_C in repositories.items():
            cheap = evaluate_cheap(
                R=R,
                loading_pct=loading_pct,
                surface_limit_C=surface_limit_C,
                properties=properties,
                cfg=cfg,
            )

            for arch_name, archetype in archetypes.items():
                gate = evaluate_gate(
                    R=R,
                    loading_pct=loading_pct,
                    archetype=archetype,
                    properties=properties,
                    cfg=cfg,
                    t_coolers_off_yr=cheap["t_coolers_off_yr"],
                )

                # The HTC the facility would have to deliver, held a degree
                # below the limit so it is a specification rather than the
                # critical value. Meaningless without the ambient it assumes,
                # which travels alongside it in the same row.
                min_h = np.nan
                if gate["Feasible"]:
                    try:
                        min_h = find_min_h_active(
                            R=R,
                            loading_fraction=loading_pct / 100.0,
                            properties=properties,
                            rho_base=rho_base,
                            cfg=cfg,
                            ambient_C=archetype["ambient_C"],
                            cooling_years=gate["t_encap_yr"],
                        )
                    except (ValueError, RuntimeError):
                        min_h = np.nan

                duty = gate["Q_per_canister_W"] * cheap["N_canisters"]

                rows.append({
                    "Name": cand["name"],
                    "Radius_m": R,
                    "Loading_Pct": loading_pct,
                    "Material": cfg["waste_form_name"],
                    "Geology": geology,
                    "Archetype": arch_name,
                    "N_canisters": cheap["N_canisters"],
                    "t_encap_yr": gate["t_encap_yr"],
                    "t_coolers_off_yr": cheap["t_coolers_off_yr"],
                    "t_active_yr": gate["t_active_yr"],
                    "t_geo_yr": cheap["t_geo_yr"],
                    "Feasible": gate["Feasible"],
                    "Binding_At_Geo": cheap["Binding_At_Geo"],
                    "T_peak_centreline_C": gate["T_peak_centreline_C"],
                    "T_peak_surface_C": gate["T_peak_surface_C"],
                    "Q_per_canister_W": gate["Q_per_canister_W"],
                    "Facility_Duty_W": duty,
                    "Min_H_Active": min_h,
                    "h_active": float(archetype["h"]),
                    "T_ambient_active_C": float(archetype["ambient_C"]),
                    "h_passive": h_passive,
                    "T_ambient_passive_C": T_passive_K - KELVIN_OFFSET,
                })

    return pd.DataFrame(rows)
