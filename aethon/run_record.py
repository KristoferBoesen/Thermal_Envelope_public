"""
A record of the settings that produced a set of results.

Results are read long after they are produced, often by someone who did not
produce them. The CSVs carry the design and its operating conditions, but not
the pre-encapsulation window, the safety factor, the campaign mass, or which
decay curve was used — so a directory of results could not previously be
identified from its own contents. Run five variations into five directories and
telling them apart depended on remembering what you typed.

Writing the resolved settings alongside the results closes that. The file is:

**Self-contained.** Decay terms and material coefficients are written in full
rather than referenced by path, so the record stays truthful even if
``waste_source.yaml`` is later regenerated from a different inventory.

**Re-runnable.** The header carries the exact command that reproduces the run,
and it needs no flags: everything, including which technologies were compared,
is in the file.

Both enumerated dimensions round-trip, but by different routes. ``repositories``
*defines* the geologies, so writing it is enough. ``cooling_archetypes`` only
*overrides* entries in the built-in library — worth keeping, since it lets a
user correct one technology's figures without restating the others — so the
selection is written separately as ``archetypes``. Without that, a rerun would
quietly widen to every technology and produce more rows than the results it
claims to describe.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

FILENAME = "run_config.yaml"


def build_run_record(
    cfg: Dict[str, Any],
    repositories: Dict[str, float],
    archetypes: Dict[str, Dict[str, Any]],
    radii: Optional[Any] = None,
    loadings_pct: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Assemble the settings of a completed run as a plain, writable mapping.

    Parameters
    ----------
    cfg : dict
        Configuration after command-line overrides were applied.
    repositories : dict
        Geology name -> surface limit [degC], as actually evaluated.
    archetypes : dict
        Cooling technologies as actually evaluated.
    radii : array-like, optional
        The radius grid used, if it differed from the config bounds.
    loadings_pct : list of float, optional
        The loadings used, if they differed from the config.

    Returns
    -------
    dict
        Ready for ``yaml.safe_dump``, and valid input to ``load_config``.
    """
    material = cfg["waste_form_name"]

    record: Dict[str, Any] = {
        "waste_source": None,
        "decay_terms": cfg["decay_terms"],
        "material": material,
        "materials": {material: dict(cfg["material_spec"])},
        "campaign": {
            "total_waste_mass_kg": cfg["total_waste_mass_kg"],
            "canister_aspect_ratio": cfg["canister_aspect_ratio"],
        },
        "pre_encapsulation_years": {
            "min": cfg["pre_encap_min_years"],
            "max": cfg["pre_encap_max_years"],
        },
        "safety_factor": cfg["safety_factor"],
        "passive": {
            "ambient_C": cfg["passive_ambient_C"],
            "h": cfg["passive_h"],
        },
        "repositories": {
            name: {"surface_limit_C": float(limit)}
            for name, limit in repositories.items()
        },
        "cooling_archetypes": {
            name: {"h": float(spec["h"]), "ambient_C": float(spec["ambient_C"])}
            for name, spec in archetypes.items()
        },
        # cooling_archetypes merges into the built-in library rather than
        # replacing it, so listing entries there does not deselect the others.
        # The selection has to be stated separately or a rerun would quietly
        # widen to every technology.
        "geologies": list(repositories),
        "archetypes": list(archetypes),
        "nodes": cfg["nodes"],
        "max_years": cfg["max_years"],
    }

    # The decay curve is inlined above, so a stale path would only mislead.
    record.pop("waste_source")

    if radii is not None and len(radii):
        record["radii"] = {
            "min": float(min(radii)),
            "max": float(max(radii)),
            "steps": int(len(radii)),
        }
    else:
        record["radii"] = {
            "min": cfg["radii_min"],
            "max": cfg["radii_max"],
            "steps": cfg["radii_steps"],
        }

    # The loading grid is written as the explicit list that ran, so an
    # arbitrary --loadings set round-trips as faithfully as a range does. The
    # range block is still required by the loader, so it is carried through.
    record["loadings"] = {
        "min": cfg["loadings_min"],
        "max": cfg["loadings_max"],
        "steps": cfg["loadings_steps"],
    }

    # The grid that actually ran is written as an explicit list, not as the
    # range that generated it: a reader should not have to reproduce a
    # linspace to know which loadings were evaluated.
    if loadings_pct is None:
        from aethon.design.search import resolve_loadings
        loadings_pct = resolve_loadings(cfg)
    record["loadings_pct"] = [float(x) for x in loadings_pct]

    if cfg.get("candidates"):
        record["candidates"] = [dict(c) for c in cfg["candidates"]]

    return record


def write_run_record(
    output_dir: Path,
    cfg: Dict[str, Any],
    repositories: Dict[str, float],
    archetypes: Dict[str, Dict[str, Any]],
    radii: Optional[Any] = None,
    loadings_pct: Optional[List[float]] = None,
) -> Path:
    """
    Write ``run_config.yaml`` into *output_dir* and return its path.

    Provenance that the configuration system cannot know — which isotope
    inventory and decay chain the curve came from — is carried through as
    comments from the ``waste_source`` file, so the chain of custody reaches
    back to the original nuclide list.
    """
    record = build_run_record(cfg, repositories, archetypes, radii, loadings_pct)

    # Everything the run needs is in the file, including which technologies
    # were selected, so the command carries no flags.
    command = f"aethon --config {FILENAME}"

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Settings that produced the results in this directory.",
        f"# Written by AETHON on {stamp}.",
        "#",
        "#   Reproduce with:",
        f"#     {command}",
        "#",
        "# Self-contained: the decay curve and material coefficients are",
        "# written out in full, not referenced, so this stays accurate even if",
        "# the files they came from are regenerated.",
    ]

    source = cfg.get("waste_source_path")
    if source:
        try:  # a short relative path reads better in a record
            source = str(Path(source).relative_to(Path.cwd()))
        except ValueError:
            pass
        lines += ["#", f"# Decay curve originally from: {source}"]

    header = "\n".join(lines) + "\n\n"
    body = yaml.safe_dump(record, sort_keys=False, default_flow_style=None)

    path = Path(output_dir) / FILENAME
    path.write_text(header + body, encoding="utf-8")
    return path
